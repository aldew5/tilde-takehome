# Note to Tilde
UPDATE: regained access Mon. afternoon and wrote a quick implementation of recursive KV (because I thought that one would be fun to write). Still needs to be tested

I focused the majority of my efforts on writing a more exploratory report which I found to be a lot more fun/interesting given the issues below.

I lost my progress and access to compute on Sat afternoon:

"The Vaughan compute cluster's GPFS storage system has gone offline. This storage system houses user home & project dirs, some data sets and model weights, . Without this system, the cluster is essentially offline.
We attempted remediation but were unsuccessful. Our next step is to open a support ticket with the vendor and follow their troubleshooting guidelines. At this time, we do not have an ETA of when the system will be back in working order."

# MLR Take home

Setup:
```bash
$ git submodule update --init --recursive
$ pip install uv
$ uv sync
$ source .venv/bin/activate
```

The repo contains a simple implementation of the K-norm filter, an eviction policy that uses the norm of the keys. You can run evaluations with the LM eval harness.

