-- =============================================================================
-- StreamHub - PostgreSQL Initialization Script
-- =============================================================================
-- This script runs automatically when the PostgreSQL container is first created
-- It sets up the pgvector extension and creates the initial schema
-- =============================================================================

-- Enable pgvector extension for vector similarity search
-- This is essential for content deduplication based on semantic similarity
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- Trigram similarity for text search

-- Create custom types for content status and source types
DO $$ BEGIN
    CREATE TYPE content_status AS ENUM (
        'pending',      -- Waiting to be processed
        'processing',   -- Currently being processed
        'processed',    -- Successfully processed
        'duplicate',    -- Identified as duplicate
        'failed',       -- Processing failed
        'archived'      -- Archived/hidden from view
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE source_type AS ENUM (
        'telegram',
        'whatsapp',
        'instagram',
        'twitter',
        'rss',
        'youtube',
        'reddit',
        'manual'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'StreamHub database initialized successfully';
    RAISE NOTICE 'pgvector extension: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;
