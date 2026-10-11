# L100EC curriculum BF16: verified split manifest

IMPORTANT: this commit contains the manifest, NOT the three binary weight parts. Do not mark GitHub checkpoint synchronization complete from the existence of this directory.

The complete checkpoint was found locally, split into raw 80,000,000 + 80,000,000 + 40,069,384-byte parts, and reconstructed. Full SHA256: `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`. It contains 135 BF16 tensors / 100,028,328 parameters. This is the inherited L100EC curriculum checkpoint, not L100QL. No new optimizer steps were performed; original optimizer and RNG were not recovered.

A separate private Drive backup stores the three real parts plus MANIFEST.json and the recovery toolkit. Access was not made public. The binary parts are also downloadable in the conversation.

The runtime's direct GitHub network path failed DNS resolution; the connector exposes inline text/base64 writes but no local binary upload parameter. GitHub binary publication remains unverified.

## Restore, after downloading all parts

Place the three parts and MANIFEST.json in one directory, then run:

```sh
python runs/artifact_recovery_20261012/checkpoint_parts.py restore PATH_TO_PARTS model.safetensors
```

## Publish real parts from an authenticated Git-enabled machine

```sh
python runs/artifact_recovery_20261012/checkpoint_parts.py publish PATH_TO_PARTS --destination models/l100ec_curriculum_bf16_07e33eb2 --author-name "FlyGraph artifact publisher" --author-email "flygraph-artifacts@users.noreply.github.com"
```

The helper validates all bytes, fills this manifest-only destination, prevents LFS/text conversion, performs a non-force push, clones again, validates the remotely stored parts and reconstructs the complete checkpoint. Only a successful remote readback counts as binary publication.
