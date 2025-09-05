# test_vit3d_full.py
import torch
from models.ViT3D import ViT3D
from AAL_APE.patch_utils import compute_patch_centers  # <— 直接导入

def main():
    B = 2
    D, H, W = 197,233,189
    img = torch.randn(B,1,D,H,W)
    affine = torch.eye(4).unsqueeze(0).expand(B,-1,-1)
    original_shape = (D, H, W)

    model = ViT3D(num_classes=4, patch_size=(16,16,16))
    model.train()

    # 调试版前向
    def forward_debug(x, mri_affine, original_shape):
        B = x.shape[0]
        # 1) embed + cls
        x_embed = model.patch_embed(x)
        cls = model.cls_token.expand(B, -1, -1)
        x1 = torch.cat([cls, x_embed], dim=1)

        # 2) AAL-APE
        centers = compute_patch_centers(
            img_shape=original_shape,
            patch_size=model.patch_embed.patch_size,
            padding=(11,7,3)
        )  # [N,3]
        centers = centers.unsqueeze(0).expand(B,-1,-1).to(x.device)
        region_pe, region_ids = model.aal_pe(centers, mri_affine, return_ids=True)
        cls_pe = torch.zeros((B,1,model.hidden_dim), device=x.device)
        x2 = x1 + torch.cat([cls_pe, region_pe], dim=1)

        # 3) 并行模块
        patch_tokens = x2[:,1:,:]
        tl_out   = model.token_learner(patch_tokens)
        ream_out = model.ream(patch_tokens, region_ids)

        print("patch_tokens:", patch_tokens.shape)
        print("TokenLearner:", tl_out.shape)
        print("REAM:", ream_out.shape)

        # 4) 拼接 + encoder
        x3 = torch.cat([x2[:,:1,:], tl_out, ream_out], dim=1)
        print("Combined:", x3.shape)
        x_enc = model.encoder_blocks(x3)
        x_enc = model.encoder_norm(x_enc)
        logits = model.heads(x_enc[:,0])
        print("Logits:", logits.shape)
        return logits

    # 替换 forward
    model.forward = forward_debug

    logits = model(img, affine, original_shape)
    loss = logits.sum()
    loss.backward()
    print("Backward OK")

if __name__=="__main__":
    main()
