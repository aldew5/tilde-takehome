# Note to Tilde
I lost my progress and access to compute on Sat afternoon. Here is the exact situation (message from admin):

"The Vaughan compute cluster's GPFS storage system has gone offline. This storage system houses user home & project dirs, some data sets and model weights, . Without this system, the cluster is essentially offline.
We attempted remediation but were unsuccessful. Our next step is to open a support ticket with the vendor and follow their troubleshooting guidelines. At this time, we do not have an ETA of when the system will be back in working order."

Restarted development on macOS without GPUs Sunday afternoon. Trying to use smaller models, etc. but it's extremely slow. Hard to measure effect of compression on models that are too small, hard to test. I focused the majority of my efforts on writing a more exploratory report which I found to be a lot more fun/interesting given the above constraints.

# MLR Take home

Setup:
```bash
$ git submodule update --init --recursive
$ pip install uv
$ uv sync
$ source .venv/bin/activate
```

The repo contains a simple implementation of the K-norm filter, an eviction policy that uses the norm of the keys. You can run evaluations with the LM eval harness.

