CREATE TABLE IF NOT EXISTS ground_truth (
    event_id        TEXT PRIMARY KEY,
    ts              DOUBLE PRECISION,
    drift_type      TEXT,
    subtype         TEXT,
    feature         TEXT,
    severity        TEXT,
    injected        BOOLEAN,
    ground_truth    TEXT
);

CREATE TABLE IF NOT EXISTS detection_events (
    id                  SERIAL PRIMARY KEY,
    event_id            TEXT REFERENCES ground_truth(event_id),
    batch_ts            DOUBLE PRECISION,
    detection_ts        TIMESTAMP DEFAULT now(),
    schema_drift_flag   BOOLEAN,
    schema_details      JSONB,
    schema_score        DOUBLE PRECISION,
    statistical_ks      DOUBLE PRECISION,
    statistical_psi     DOUBLE PRECISION,
    statistical_details JSONB,
    statistical_score   DOUBLE PRECISION,
    semantic_drift_flag BOOLEAN,
    semantic_details    JSONB,
    semantic_score      DOUBLE PRECISION,
    unified_score       DOUBLE PRECISION,
    decision            TEXT
);

CREATE TABLE IF NOT EXISTS drift_events (
    id              SERIAL PRIMARY KEY,
    event_id        TEXT REFERENCES ground_truth(event_id),
    detected_type   TEXT,
    severity_score  DOUBLE PRECISION,
    decision        TEXT,
    detection_ts    TIMESTAMP DEFAULT now()
);