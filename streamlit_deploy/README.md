# Glance-Focus ViT demo

Streamlit image demo using the seed 42 `best_v2` EMA checkpoint and Grad-CAM explanations. The app is for research and demonstration, not clinical diagnosis.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and copy this folder's contents into its root.
2. Install Git LFS, then run `git lfs install` and `git lfs track "*.pt"`.
3. Commit and push the repository. The checkpoint is stored with Git LFS.
4. In Streamlit Community Cloud, create an app from that repository with `streamlit_app.py` as the entrypoint. Keep the repository/app private if access should be restricted.

The checkpoint uses about 350 MB after removing optimizer state and the unused raw weights. Community Cloud resource limits vary; this ViT-B/16 app may need more than the minimum memory allocation.
