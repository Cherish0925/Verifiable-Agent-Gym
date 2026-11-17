"""A small scripted adapter example; replace the chosen SQL with a model policy."""

import argparse
from pathlib import Path

from agent_gym.episode import Episode
from agent_gym.taskpacks import load_taskpack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = load_taskpack(Path(__file__).with_name("tasks.jsonl"), family="warehouse-sql")
    # The first training task is the revenue question. The policy uses a distinct
    # query formulation and only the exposed instruction/schema/query interface.
    with Episode(pack.make(pack.specs("train")[0]), args.output, agent="sql-example-policy") as episode:
        print(episode.initial["prompt"])
        print(episode.step({"tool": "database_schema", "arguments": {}}))
        print(episode.step({"tool": "query", "arguments": {"sql": "SELECT status,COUNT(*) FROM orders GROUP BY status"}}))
        result = episode.step({"submit": {"sql": "WITH bills AS (SELECT order_id,SUM(quantity*unit_price) revenue FROM order_items GROUP BY order_id) SELECT region,SUM(revenue) total FROM bills JOIN orders USING(order_id) JOIN customers USING(customer_id) WHERE status='completed' AND ordered_at BETWEEN '2026-10-01' AND '2026-10-31' GROUP BY region ORDER BY total DESC,region"}})
        print(result["result"]["status"], result["result"]["reward"])


if __name__ == "__main__":
    main()
