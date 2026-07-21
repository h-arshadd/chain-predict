from crypto_pipeline.utils.metadata_utils import (
    get_db_connection,
    create_all_metadata_tables,
    load_strategies_from_yaml,
)

conn = get_db_connection()

create_all_metadata_tables(conn)  # recreates metadata.strategy fresh, with UNIQUE constraint built in

load_strategies_from_yaml(conn, "crypto_pipeline/signals/strategies")

conn.close()