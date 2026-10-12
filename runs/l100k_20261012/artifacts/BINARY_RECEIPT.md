# L100K verified actual native checkpoint bytes

These 3 binary Git blobs contain **real model.safetensors and optimizer/RNG training_state.pt** from the completed L100K pair. All are ordinary Git objects, not Git LFS pointers. These are narrow neural memory modules; **the 100M generative base was NOT fine-tuned by L100K**.

- `all_native_checkpoints.tar.xz`: 1,427,576 bytes; SHA256 `851cb1e254423310de3bf79f39dbfe7900808dccfb63d4310d562463f1b80e29`; Git blob `a6201e56179f2ea416a52b778ced46a2d31dd082`. Contains optimizer/RNG, model bytes and manifest at **every 64 steps through step 1024 for both arms**.
- `source_data_results.tar.xz`: 83,104 bytes; SHA256 `7f9b4b94fa8a9078ea64faafa2c7db30d2d9d8dc22a5806b87b5cab699c461e2`; blob `a0fd212f074e392e9c730c2c980b8760519ea150`.
- `final_native_checkpoints.tar.xz`: 108,732 bytes; SHA256 `2da64fd5a0b11a7daab1c18712c4fe203010b04bdd4ce35362fc087db333d4de`; blob `0678da9dbaaa979a1489ff6e46180019cf2e71c6`.

Restoration: `tar -xJf artifacts/source_data_results.tar.xz`, `tar -xJf artifacts/all_native_checkpoints.tar.xz`. Use safe extraction on trusted bytes; the source archive includes harmless cached files from the executed environment, which are not required. Then verify per-checkpoint `MANIFEST.json` hashes and run tests.

Fixed held outcome: IID query-aware **200/200 both-correct pairs** vs query-blind **0/200**, unseen letters **54/100** vs **0/100**. Strict narrow ASCII byte memory with an engineered copy operator; not general natural language or Qwen comparison.

These blobs were created using the actual recovered archive bytes. Their Git blob SHA1 identifiers matched the independently computed local Git-blob SHA1 for each binary before tree creation.
