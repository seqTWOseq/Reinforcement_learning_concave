초기 설정

```
cd c:\tls_dyd\Tlsdyd\Omoc\Reinforcement_learning_concave
python -m pip install -r requirements.txt
```

PPO vs Heuristic

```
python gomoku_project/scripts/train_ppo_vs_heuristic.py --games 100
```

PPO vs AlphaZero

```
python gomoku_project/scripts/train_ppo_vs_alphazero.py --games 10
```

PPO vs Human

```
python gomoku_project/scripts/train_ppo_vs_human.py
```

AlphaZero vs Heuristic

```
python gomoku_project/scripts/train_alphazero_vs_heuristic.py --games 10
```

AlphaZero vs Human

```
python gomoku_project/scripts/train_alphazero_vs_human.py
```

Heuristic vs Human

```
python gomoku_project/scripts/play_human_vs_ai.py --opponent heuristic --human-color black
```
