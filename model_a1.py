"""GF-ViT Level 1 A1 model with evaluation-time diagnostics.

This module keeps the project's existing ViT, ROI, gate, dropout, head and
auxiliary-loss implementations. It adds a diagnostic forward that can expose
both auxiliary classifiers while the model stays in eval mode.
"""
import torch
from torch.nn import functional as F

from model import GlanceFocusViT, ModelEMA


class GFViTLevel1A1(GlanceFocusViT):
    """A1.1 Global, A1.2 Local, or A1.3 shared-backbone Gate model.

    ``forward`` preserves the normal training interface and existing local
    branch dropout. ``forward_diagnostics`` is read-only and returns auxiliary
    logits, features, ROI, gate and local-contribution statistics in eval mode.
    """

    def _classify(self, features):
        if self.legacy:
            return self.head(features)
        return self.head(self.drop(features))

    def forward(self, hi, mask=None, selector=None, diagnostics=False):
        if diagnostics and self.training:
            raise RuntimeError('Diagnostics require model.eval()')
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W]')
        if self.use_fov and self.fusion != 'global':
            if mask is None or mask.shape != (hi.shape[0], 1, *hi.shape[-2:]):
                raise ValueError('use_fov requires mask [B,1,H,W] matching hi')
        if self.fusion == 'local' and selector is None:
            raise ValueError('A1.2 requires its frozen A1.1 selector')
        if selector is not None:
            if selector is self:
                raise ValueError('Selector must be a separate model, not self')
            if selector.training:
                raise ValueError('Selector must be in eval mode')
            if any(parameter.requires_grad for parameter in selector.parameters()):
                raise ValueError('Selector parameters must be frozen')

        global_view = F.interpolate(
            hi, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        global_view = (global_view - .5) / .5

        if self.fusion == 'global':
            z_global = self.vit(global_view)
            if not self.legacy and self.view_embed:
                z_global = z_global + self.view_g
            logits = self._classify(z_global)
            output = {
                'logits': logits, 'boxes': None, 'gate': None,
                'aux_g': None, 'aux_l': None, 'attention': None,
            }
            if diagnostics:
                output.update(logits_global=logits, z_global=z_global)
            return output

        if selector is not None:
            with torch.no_grad():
                _, attention = selector.encode_map(global_view)
            z_global = self.vit(global_view) if self.fusion != 'local' else None
        else:
            z_global, attention = self.encode_map(global_view)

        boxes, attention = self.select_boxes(attention, mask, *hi.shape[-2:])
        # Boxes use high-resolution pixel-edge coordinates (x1,y1,x2,y2).
        t = (torch.arange(224, device=hi.device, dtype=hi.dtype) + .5) / 224
        xs = boxes[:, 0, None] + (boxes[:, 2] - boxes[:, 0])[:, None] * t
        ys = boxes[:, 1, None] + (boxes[:, 3] - boxes[:, 1])[:, None] * t
        gx = (2 * xs / hi.shape[-1] - 1)[:, None, :].expand(-1, 224, -1)
        gy = (2 * ys / hi.shape[-2] - 1)[:, :, None].expand(-1, -1, 224)
        local_view = F.grid_sample(
            hi, torch.stack([gx, gy], -1), align_corners=False, padding_mode='border')
        z_local = self.vit((local_view - .5) / .5)

        if not self.legacy and self.view_embed:
            if z_global is not None:
                z_global = z_global + self.view_g
            z_local = z_local + self.view_l

        gate = aux_g = aux_l = None
        delta_local = None
        local_contrib_ratio = None
        if self.fusion == 'local':
            fused_features = z_local
        else:
            if not self.legacy and self.training and self.branch_drop > 0:
                keep = (torch.rand(z_local.shape[0], 1, device=z_local.device,
                                   dtype=z_local.dtype) >= self.branch_drop)
                z_local_used = z_local * keep
            else:
                z_local_used = z_local

            if self.fusion == 'concat':
                fused_features = self.ln_cat(torch.cat([z_global, z_local_used], -1))
            else:
                gate = self.gate(torch.cat(
                    [self.ln_g(z_global), self.ln_l(z_local_used)], -1))
                delta_local = self.s * gate * z_local_used
                fused_features = z_global + delta_local
                local_contrib_ratio = (
                    delta_local.norm(dim=1) / (z_global.norm(dim=1) + 1e-8))

            need_aux = self.training or diagnostics
            if not self.legacy and need_aux:
                aux_g = self.aux_g(z_global)
                aux_l = self.aux_l(z_local)

        logits = self._classify(fused_features)
        output = {
            'logits': logits, 'boxes': boxes, 'gate': gate,
            'aux_g': aux_g, 'aux_l': aux_l, 'attention': attention,
            'local_contrib_ratio': local_contrib_ratio,
        }
        if diagnostics:
            output.update(z_global=z_global, z_local=z_local)
            if self.fusion == 'gate':
                # Eval mode disables local-branch dropout, so this is the
                # exact local term used by the main fused prediction.
                delta_local = self.s * gate * z_local
                logits_global_only = self._classify(z_global)
                logits_fused = self._classify(z_global + delta_local)
                output.update(
                    logits_fused=logits,
                    logits_aux_global=aux_g,
                    logits_aux_local=aux_l,
                    logits_global_only=logits_global_only,
                    logit_delta=logits_fused - logits_global_only,
                    local_contrib_ratio=(
                        delta_local.norm(dim=1) / (z_global.norm(dim=1) + 1e-8)),
                    s=self.s.detach().clone(),
                    local_delta=delta_local,
                )
        return output

    @torch.no_grad()
    def forward_diagnostics(self, hi, mask=None, selector=None):
        """Run inference diagnostics without changing the model's train/eval state."""
        if self.training:
            raise RuntimeError('Call model.eval() before forward_diagnostics()')
        return self.forward(hi, mask, selector, diagnostics=True)


__all__ = ['GFViTLevel1A1', 'ModelEMA']
