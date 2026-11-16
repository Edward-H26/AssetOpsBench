# examples

The smallest use of the environment, without a model and without the tool servers.

```bash
python -m CausalDiagnosis examples run-one-episode --seed 0
```

`run_one_episode.py` plays one episode of the oracle policy and one of the do-nothing policy through the Gymnasium wrapper and prints which of the two earned the higher reward. It needs the data file of hyd_1, which `python -m CausalDiagnosis datasets download hyd_1 --download` builds.

A new example is one short program here, started as a command of the group `examples`.
