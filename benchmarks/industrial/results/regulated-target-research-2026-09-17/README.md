# Public target research artifact

This is a structural dataset audit, not a model or evaluator benchmark result.
Four official TAT-QA JSON files were fetched at commit
`870accc41953dcde885aabeb963d94aabdc0fbc3` on September 17, 2026.
`tatqa-audit.json` records their SHA-256 hashes and exact counting/matching rules.
No target or judge API calls were made.

Reproduce with Python 3.10 or newer, from the repository root:

```bash
python benchmarks/industrial/audit_public_financial_data.py --cache /tmp/tatqa-audit --download
```

Compare the produced JSON with the committed artifact. Omit `--download` to
repeat the audit offline on the cached files. The audit was run with Python
3.10 and 3.12 and produced identical JSON. Raw source documents and labels are
not redistributed here; obtain the pinned files from their original project.

Attribution: Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang,
Jiancheng Lv, Fuli Feng and Tat-Seng Chua, *TAT-QA: A Question Answering Benchmark
on a Hybrid of Tabular and Textual Content in Finance*, ACL 2021.
[Official repository](https://github.com/NExTplusplus/TAT-QA) and
[paper](https://aclanthology.org/2021.acl-long.254/).
The upstream README states CC BY 4.0 for the dataset.

The [study](../../../../plans/regulated-enterprise-study.md) explains the
test/test-gold identifier mismatch, exact-content differences and limits of
context IDs as evidence of source independence.
