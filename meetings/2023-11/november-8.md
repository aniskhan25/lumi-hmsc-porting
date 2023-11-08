# 8 November, 2023 Meeting Notes

-----

**Attendees:**

| Name             | Organization |
| ---------------- | ------------ |
| Anis U. Rahman   | JYU,FI       |
| Gleb Tikhonov    | UH,FI        |
| Tuomas Rossi     | CSC,FI       |
| Tim Dykes        | HPE,UK       |
| Rafael Sarmiento | CSCS,CH      |

### Topic

- Finalisation of project logistics

### Summary

- Next steps on project management
- Multi-gpu implementation
	- Code walk-thru and demo on Lumi
	- Discussion on anamolous ayncio performance

### Conclusion

- Mantain a separate repository for porting program
- Further exploration needed for multi-gpu implementation, identify alternative multithreading packages/approaches

### To-dos

- Create repo (Anis)
- Profile asyncio multi-gpu implementation (Anis)
- Attempt to implement multi-gpu using MPI (Anis)
- Share resources on project management (Tim)
- Profile single-gpu implementation on Lumi for identification of AMD-specific bottlenecks (Gleb)
