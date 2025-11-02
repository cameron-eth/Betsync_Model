-- Create NBA predictions table for storing model predictions
-- Run this in Supabase SQL editor

CREATE TABLE IF NOT EXISTS nba_predictions (
    id BIGSERIAL PRIMARY KEY,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    game_date DATE NOT NULL,
    home_win_probability FLOAT,
    away_win_probability FLOAT,
    predicted_winner VARCHAR(255),
    expected_home_score FLOAT,
    expected_away_score FLOAT,
    expected_total FLOAT,
    confidence_score FLOAT,
    confidence_label VARCHAR(50),
    prediction_timestamp TIMESTAMP,
    model_version VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicate predictions for same game
    CONSTRAINT unique_nba_game UNIQUE(home_team, away_team, game_date)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_nba_predictions_game_date ON nba_predictions(game_date);
CREATE INDEX IF NOT EXISTS idx_nba_predictions_teams ON nba_predictions(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_nba_predictions_created_at ON nba_predictions(created_at);

-- Enable Row Level Security (optional)
ALTER TABLE nba_predictions ENABLE ROW LEVEL SECURITY;

-- Create policy to allow public read access (optional - adjust as needed)
CREATE POLICY "Allow public read access" ON nba_predictions
    FOR SELECT
    USING (true);

-- Create policy to allow authenticated inserts/updates (optional - adjust as needed)
CREATE POLICY "Allow authenticated write access" ON nba_predictions
    FOR ALL
    USING (auth.role() = 'authenticated');

