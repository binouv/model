# One-shot transfer of existing checkpoint bytes

This job does NOT train any model. It only imports the user's already-verified L100EC checkpoint parts into this public repository and verifies a fresh remote clone.

The runner generates an ephemeral RSA-4096 key outside the checkout and publishes only its public key. The sender encrypts short-lived artifact URLs with RSA-OAEP/SHA256. Each encrypted block binds transfer nonce, ordinal, block count and full payload hash. No bearer URL or private key is written in Git, workflow logs, caches or artifacts. The authenticated repository write path authorizes the ciphertext handoff; encryption is not claimed as a standalone sender signature.

The job accepts exactly the three precommitted part names, sizes and SHA256 hashes. HTTPS downloads are restricted to the artifact delivery host; redirects and byte counts are checked. After transfer it makes a normal non-force Git push, clones again via HTTPS, rehashes every remotely stored part and reconstructs the exact checkpoint. Only then is GITHUB_BINARY_RECEIPT.json published. Public-key and receipt commits skip CI, and the workflow triggers only on REQUEST.json.

The handoff wait is bounded; the private key exists only in the job's temporary directory and is removed on normal or handled-failure exit. URLs are already independently scoped and short-lived. ABORT.json in this run directory aborts a waiting job. Existing L100QL/L100V/L100CS research is untouched.

11 transport tests and 19 checkpoint tests passed locally before registration. No successful GitHub binary publication is claimed by this source commit; inspect the final receipt and workflow conclusion.
