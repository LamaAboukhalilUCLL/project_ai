CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    display_name    TEXT,
    location        TEXT,
    reputation      INTEGER,
    creation_date   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posts (
    id              INTEGER PRIMARY KEY,
    post_type_id    INTEGER,
    owner_user_id   INTEGER,
    title           TEXT,
    score           INTEGER,
    view_count      INTEGER,
    answer_count    INTEGER,
    creation_date   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS votes (
    id              INTEGER PRIMARY KEY,
    post_id         INTEGER,
    vote_type_id    INTEGER,
    creation_date   TIMESTAMP
);

-- Enable pg_stat_statements so the monitor can read query stats later
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;