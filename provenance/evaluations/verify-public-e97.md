# Public E97 Checkpoint URL Verification

Task: `verify-public-e97`
Verified at: 2026-07-29T14:43:26Z

## Authoritative Object

S3 URI:
`s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`

Virtual-hosted HTTPS URL:
`https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`

Expected sha256 from handoff:
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

The sha256 was not revalidated because the task explicitly prohibited
downloading the checkpoint body.

## Anonymous HEAD Result

Command shape:
`curl --head --location --max-redirs 10 --connect-timeout 10 --max-time 30`

No signed query parameters, credentials, or authentication headers were used.
The request completed as a HEAD-only probe with `CURL_SIZE_DOWNLOAD:0`.

- HTTP status: `200 OK`
- Redirects: `0`
- Final endpoint: `https://spinozans.s3.amazonaws.com/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- Content-Length: `7719680116`
- Content-Type: `application/vnd.snesdev-page-table`
- ETag: `"f3f88f4a11fad751ee203baa5c10822f-116"`
- Last-Modified: `Wed, 22 Jul 2026 08:03:56 GMT`
- Server: `AmazonS3`

Anonymous public HTTPS access is verified for the exact bucket/key above.
