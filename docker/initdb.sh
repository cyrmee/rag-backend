#!/bin/sh
# Builds a fresh database: schema.sql, then every numbered migration in
# order. Postgres runs this once, only when its data volume is empty (an
# existing database is never touched). The SQL files are mounted at /sql
# rather than straight into /docker-entrypoint-initdb.d, which would run
# them alphabetically - 002_... before schema.sql.
set -e
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /sql/schema.sql
for migration in $(ls /sql/[0-9][0-9][0-9]_*.sql | sort); do
    psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$migration"
done
