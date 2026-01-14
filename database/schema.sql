-- ============================================================================
-- StreamHub Content Aggregation Platform - Database Schema
-- PostgreSQL with pgvector extension for vector embeddings
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- For UUID generation
CREATE EXTENSION IF NOT EXISTS "pgvector";       -- For vector similarity search

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Source types for content aggregation
CREATE TYPE source_type AS ENUM (
    'telegram',
    'whatsapp',
    'instagram',
    'twitter',
    'rss',
    'youtube',
    'reddit'
);
COMMENT ON TYPE source_type IS 'Supported content source platforms';

-- Content types for unified storage
CREATE TYPE content_type AS ENUM (
    'text',
    'image',
    'video',
    'link'
);
COMMENT ON TYPE content_type IS 'Types of content that can be aggregated';

-- User roles for access control
CREATE TYPE user_role AS ENUM (
    'admin',
    'editor',
    'viewer'
);
COMMENT ON TYPE user_role IS 'User permission levels for dashboard access';

-- ============================================================================
-- SOURCES TABLE
-- Tracks configured content sources for aggregation
-- ============================================================================

CREATE TABLE sources (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    source_type     source_type NOT NULL,
    config          JSONB NOT NULL DEFAULT '{}',
    enabled         BOOLEAN NOT NULL DEFAULT true,
    last_sync       TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT sources_name_unique UNIQUE (name)
);

COMMENT ON TABLE sources IS 'Configured content sources for aggregation (Telegram channels, RSS feeds, etc.)';
COMMENT ON COLUMN sources.config IS 'Source-specific configuration (API keys, channel IDs, feed URLs, etc.)';
COMMENT ON COLUMN sources.last_sync IS 'Timestamp of the last successful content sync';

-- Indexes for sources
CREATE INDEX idx_sources_type ON sources(source_type);
CREATE INDEX idx_sources_enabled ON sources(enabled) WHERE enabled = true;
CREATE INDEX idx_sources_last_sync ON sources(last_sync);

-- ============================================================================
-- CONTENT TABLE
-- Unified storage for all aggregated content with vector embeddings
-- ============================================================================

CREATE TABLE content (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id           INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id         VARCHAR(512) NOT NULL,
    content_type        content_type NOT NULL,
    title               TEXT,
    body                TEXT,
    author              VARCHAR(255),
    url                 TEXT,
    media_urls          JSONB DEFAULT '[]',
    embedding           vector(1536),
    importance_score    SMALLINT DEFAULT 50,
    is_duplicate        BOOLEAN NOT NULL DEFAULT false,
    raw_data            JSONB DEFAULT '{}',
    published_at        TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT content_source_external_unique UNIQUE (source_id, external_id),
    CONSTRAINT content_importance_score_range CHECK (importance_score >= 0 AND importance_score <= 100)
);

COMMENT ON TABLE content IS 'Unified storage for all aggregated content from various sources';
COMMENT ON COLUMN content.external_id IS 'Unique identifier from the source platform (tweet ID, message ID, etc.)';
COMMENT ON COLUMN content.embedding IS 'Vector embedding (1536 dimensions for OpenAI ada-002) for semantic similarity and deduplication';
COMMENT ON COLUMN content.importance_score IS 'Calculated importance score (0-100) based on engagement, relevance, etc.';
COMMENT ON COLUMN content.is_duplicate IS 'Flag indicating if this content is a duplicate of existing content';
COMMENT ON COLUMN content.raw_data IS 'Original raw data from the source platform for reference';
COMMENT ON COLUMN content.media_urls IS 'Array of media URLs (images, videos) associated with this content';

-- Indexes for content
CREATE INDEX idx_content_source_id ON content(source_id);
CREATE INDEX idx_content_type ON content(content_type);
CREATE INDEX idx_content_published_at ON content(published_at DESC);
CREATE INDEX idx_content_created_at ON content(created_at DESC);
CREATE INDEX idx_content_importance ON content(importance_score DESC);
CREATE INDEX idx_content_is_duplicate ON content(is_duplicate) WHERE is_duplicate = false;
CREATE INDEX idx_content_author ON content(author) WHERE author IS NOT NULL;

-- Vector similarity search index using IVFFlat for approximate nearest neighbor
-- Lists parameter should be tuned based on data size (sqrt(n) is a good starting point)
CREATE INDEX idx_content_embedding ON content USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- CATEGORIES TABLE
-- Hierarchical categories for content organization
-- ============================================================================

CREATE TABLE categories (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    slug            VARCHAR(255) NOT NULL UNIQUE,
    description     TEXT,
    parent_id       INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE categories IS 'Hierarchical categories for organizing and filtering content';
COMMENT ON COLUMN categories.slug IS 'URL-friendly identifier for the category';
COMMENT ON COLUMN categories.parent_id IS 'Reference to parent category for hierarchical structure';
COMMENT ON COLUMN categories.sort_order IS 'Display order within the same hierarchy level';

-- Indexes for categories
CREATE INDEX idx_categories_parent ON categories(parent_id);
CREATE INDEX idx_categories_slug ON categories(slug);
CREATE INDEX idx_categories_sort ON categories(sort_order);

-- ============================================================================
-- TAGS TABLE
-- User-defined tags for flexible content labeling
-- ============================================================================

CREATE TABLE tags (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    slug            VARCHAR(100) NOT NULL UNIQUE,
    color           VARCHAR(7) DEFAULT '#6B7280',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE tags IS 'User-defined tags for flexible content labeling and filtering';
COMMENT ON COLUMN tags.slug IS 'URL-friendly identifier for the tag';
COMMENT ON COLUMN tags.color IS 'Hex color code for visual display in the UI';

-- Indexes for tags
CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_tags_slug ON tags(slug);

-- ============================================================================
-- CONTENT_CATEGORIES JUNCTION TABLE
-- Many-to-many relationship between content and categories
-- ============================================================================

CREATE TABLE content_categories (
    content_id      UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    category_id     INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    confidence      DECIMAL(5,4) DEFAULT 1.0,
    assigned_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (content_id, category_id),
    CONSTRAINT content_categories_confidence_range CHECK (confidence >= 0 AND confidence <= 1)
);

COMMENT ON TABLE content_categories IS 'Junction table linking content to categories';
COMMENT ON COLUMN content_categories.confidence IS 'LLM classification confidence score (0-1)';

-- Indexes for content_categories
CREATE INDEX idx_content_categories_content ON content_categories(content_id);
CREATE INDEX idx_content_categories_category ON content_categories(category_id);
CREATE INDEX idx_content_categories_confidence ON content_categories(confidence DESC);

-- ============================================================================
-- CONTENT_TAGS JUNCTION TABLE
-- Many-to-many relationship between content and tags
-- ============================================================================

CREATE TABLE content_tags (
    content_id      UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    tag_id          INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    assigned_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (content_id, tag_id)
);

COMMENT ON TABLE content_tags IS 'Junction table linking content to user-defined tags';

-- Indexes for content_tags
CREATE INDEX idx_content_tags_content ON content_tags(content_id);
CREATE INDEX idx_content_tags_tag ON content_tags(tag_id);

-- ============================================================================
-- USERS TABLE
-- Dashboard users with authentication and role-based access
-- ============================================================================

CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    full_name       VARCHAR(255),
    role            user_role NOT NULL DEFAULT 'viewer',
    is_active       BOOLEAN NOT NULL DEFAULT true,
    last_login      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE users IS 'Dashboard users with authentication credentials and roles';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password';
COMMENT ON COLUMN users.role IS 'User permission level (admin, editor, viewer)';

-- Indexes for users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;

-- ============================================================================
-- USER_PREFERENCES TABLE
-- Personalization settings for dashboard users
-- ============================================================================

CREATE TABLE user_preferences (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    preferences     JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE user_preferences IS 'User-specific personalization settings stored as JSONB';
COMMENT ON COLUMN user_preferences.preferences IS 'JSON object containing theme, notification settings, dashboard layout, favorite categories, etc.';

-- Indexes for user_preferences
CREATE INDEX idx_user_preferences_user ON user_preferences(user_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at trigger to all tables with updated_at column
CREATE TRIGGER trigger_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_content_updated_at
    BEFORE UPDATE ON content
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_categories_updated_at
    BEFORE UPDATE ON categories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VECTOR SIMILARITY SEARCH FUNCTION
-- Helper function for finding similar content using embeddings
-- ============================================================================

CREATE OR REPLACE FUNCTION find_similar_content(
    query_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.85,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    content_id UUID,
    title TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.title,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM content c
    WHERE c.embedding IS NOT NULL
      AND c.is_duplicate = false
      AND 1 - (c.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_similar_content IS 'Find content similar to a given embedding vector using cosine similarity';

-- ============================================================================
-- INITIAL DATA
-- Default categories and admin user
-- ============================================================================

-- Insert default top-level categories
INSERT INTO categories (name, slug, description, sort_order) VALUES
    ('News', 'news', 'Breaking news and current events', 1),
    ('Technology', 'technology', 'Tech news, software, and gadgets', 2),
    ('Business', 'business', 'Business, finance, and markets', 3),
    ('Entertainment', 'entertainment', 'Movies, music, and pop culture', 4),
    ('Sports', 'sports', 'Sports news and updates', 5),
    ('Science', 'science', 'Scientific discoveries and research', 6),
    ('Politics', 'politics', 'Political news and analysis', 7),
    ('Health', 'health', 'Health, medicine, and wellness', 8),
    ('Other', 'other', 'Uncategorized content', 99);

-- Insert default tags
INSERT INTO tags (name, slug, color) VALUES
    ('Important', 'important', '#EF4444'),
    ('Trending', 'trending', '#F59E0B'),
    ('Breaking', 'breaking', '#DC2626'),
    ('Featured', 'featured', '#8B5CF6'),
    ('Archived', 'archived', '#6B7280');
