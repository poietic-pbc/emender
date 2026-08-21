# GDN2 Hypervolume Publish Record - 2026-07-22

Task: `publish-gdn2-plot-2`

## Published Artifact

- Local source: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/gdn2_mlp_early_progress_tokens_smoothed_loss_20260722T0916Z.png`
- Final SSH path: `erik@hypervolu.me:www/emender/gdn2_mlp_diloco_loss_curve_20260722.png`
- Public URL: `http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png`
- Expected/local SHA-256: `b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333`
- Final remote SHA-256: `b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333`
- Final HTTP SHA-256: `b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333`
- Final HTTP status/content type: `200`, `image/png`
- Final HTTP content length: `147865`

## Source Validation

The local source existed and was readable before upload.

```text
-rw-rw-r-- 1 erikg erikg 147865 Jul 22 09:17 /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/gdn2_mlp_early_progress_tokens_smoothed_loss_20260722T0916Z.png
image/png
b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333
```

## Collision Handling

The GDN2 final target was checked before upload and was absent:

```text
ssh erik@hypervolu.me "if [ -e 'www/emender/gdn2_mlp_diloco_loss_curve_20260722.png' ]; then echo exists; sha256sum 'www/emender/gdn2_mlp_diloco_loss_curve_20260722.png'; stat -c 'size=%s mtime=%y mode=%a path=%n' 'www/emender/gdn2_mlp_diloco_loss_curve_20260722.png'; else echo absent; fi"
absent
```

The upload used a same-directory temporary file:

```text
www/emender/.gdn2_mlp_diloco_loss_curve_20260722.png.tmp.publish-gdn2-plot-2.20260722T104457Z.3868556
```

The temporary file was verified before rename:

```text
b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333  www/emender/.gdn2_mlp_diloco_loss_curve_20260722.png.tmp.publish-gdn2-plot-2.20260722T104457Z.3868556
size=147865 mtime=2026-07-22 09:17:45.000000000 +0000 mode=664 path=www/emender/.gdn2_mlp_diloco_loss_curve_20260722.png.tmp.publish-gdn2-plot-2.20260722T104457Z.3868556
image/png
```

After confirming the final path was still absent, the temporary file was renamed with `mv` to the final GDN2 path. No command wrote to `www/emender/e97_diloco_loss_curve_20260623.png`.

## Final GDN2 Verification

```text
b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333  www/emender/gdn2_mlp_diloco_loss_curve_20260722.png
size=147865 mtime=2026-07-22 09:17:45.000000000 +0000 mode=664 path=www/emender/gdn2_mlp_diloco_loss_curve_20260722.png
image/png

http_code=200 content_type=image/png size_download=147865
b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333  /tmp/gdn2_http_after_publish_gdn2.png
HTTP/1.1 200 OK
Last-Modified: Wed, 22 Jul 2026 09:17:45 GMT
ETag: "24199-6572f9cbd6040"
Content-Length: 147865
Content-Type: image/png
```

## E97 Preservation Checks

The existing E97 artifact was checked before and after publishing GDN2.

Remote before:

```text
4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc  www/emender/e97_diloco_loss_curve_20260623.png
size=290596 mtime=2026-07-07 14:43:49.418248931 +0000 mode=664 path=www/emender/e97_diloco_loss_curve_20260623.png
image/png
```

HTTP before:

```text
4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc  /tmp/e97_http_before_publish_gdn2.png
HTTP/1.1 200 OK
Last-Modified: Tue, 07 Jul 2026 14:43:49 GMT
ETag: "46f24-656066b401508"
Content-Length: 290596
Content-Type: image/png
```

Remote after:

```text
4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc  www/emender/e97_diloco_loss_curve_20260623.png
size=290596 mtime=2026-07-07 14:43:49.418248931 +0000 mode=664 path=www/emender/e97_diloco_loss_curve_20260623.png
image/png
```

HTTP after:

```text
http_code=200 content_type=image/png size_download=290596
4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc  /tmp/e97_http_after_publish_gdn2.png
HTTP/1.1 200 OK
Last-Modified: Tue, 07 Jul 2026 14:43:49 GMT
ETag: "46f24-656066b401508"
Content-Length: 290596
Content-Type: image/png
```

The E97 remote hash and HTTP hash were unchanged:

```text
remote before/after: 4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc / 4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc
HTTP before/after:   4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc / 4f32c4b30f54e935c20d5669a9e26ffd2d2194f3242a77ac0efe7bb44e5968dc
```

No training, plotting, or S3 commands were run for this publish task.
