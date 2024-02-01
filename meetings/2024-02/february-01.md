# 1 February, 2024 Meeting Notes

-----

**Attendees:**

| Name             | Organization |
| ---------------- | ------------ |
| Tim Dykes        | HPE,UK       |
| Gleb Tikhonov    | UH,FI        |
| Tuomas Rossi     | CSC,FI       |
| Anis U. Rahman   | JYU,FI       |


### Topic

- Benchmarking updates

### Summary

- Benchmarking cholesky on AMD shows 4x slower performance compared to Nvidia GPU
- Explore a distributed linear algebra effort with AMD-specific implementations, DLFA-Future (https://github.com/eth-cscs/DLA-Future)
- Possible to use DLFA-Future C++ implementation as a new TF op (https://www.tensorflow.org/guide/create_op)
- Discussion on the TF on Windows with GPU, 
	- TF support via wsl2, a virtualization technology to run linux kernel within a lightweight virtual machine
	- Reasons for degraded performance on laptop with GPU RTX 4000 Ada
	- Limited DP performance of the card, possibly designed for gaming (abundant SP ops) rather than DP compute (case for Hmsc-HPC)

### To-dos

- Benchmark DLA-Future cholesky on GPU
- Implement and custom C++ op and python wrapper