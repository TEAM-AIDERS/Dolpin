# Dolpin: Requirements Document

## Introduction

This system provides real-time analysis of social media spikes and trends for entertainment industry crisis management and opportunity detection. The system ingests data from multiple sources (Twitter, community sites, Google Trends), analyzes sentiment and causality, assesses legal risks, and generates actionable executive briefs for decision-makers.

## Glossary

- **System**: The Social Media Spike Analysis System
- **Spike_Analyzer**: Component that detects and analyzes sudden increases in social media activity
- **Sentiment_Agent**: Component that classifies emotional tone of messages
- **Causality_Agent**: Component that identifies network patterns and influence sources
- **Legal_RAG_Agent**: Component that assesses legal risks using keyword matching and RAG retrieval
- **Playbook_Agent**: Component that generates strategic response recommendations
- **Router**: Decision component that determines workflow paths based on analysis results
- **Kafka_Consumer**: Component that receives messages from Kafka message queue
- **Spike_Event**: A detected sudden increase in mentions of a keyword
- **Executive_Brief**: Final output report for decision-makers
- **Actionability_Score**: Metric (0.0-1.0) indicating urgency of response needed
- **Confidence**: Metric (0.0-1.0) indicating reliability of analysis results
- **RAG**: Retrieval-Augmented Generation for legal document search
- **Trace_ID**: Unique identifier for tracking a workflow execution

## Requirements

### Requirement 1: Kafka Message Ingestion

**User Story:** As a system operator, I want to ingest real-time social media data from Kafka, so that the system can analyze current trends and spikes.

#### Acceptance Criteria

1. WHEN a Kafka message arrives, THE Kafka_Consumer SHALL parse the message into a structured format
2. WHEN the message type is "post", THE Kafka_Consumer SHALL extract content_data including text, author_id, and metrics
3. WHEN the message type is "trend", THE Kafka_Consumer SHALL extract trend_data including interest_score, rising_queries, and region_stats
4. THE Kafka_Consumer SHALL support messages from sources: "twitter", "theqoo", "instiz", "google_trends"
5. WHEN a message is malformed, THE Kafka_Consumer SHALL log the error and continue processing
6. THE Kafka_Consumer SHALL preserve the original message_id as UUID format
7. THE Kafka_Consumer SHALL record collected_at timestamp in ISO 8601 format

### Requirement 2: Spike Detection and Analysis

**User Story:** As an analyst, I want to detect significant spikes in social media activity, so that I can identify emerging issues or opportunities early.

#### Acceptance Criteria

1. WHEN messages are aggregated by keyword, THE Spike_Analyzer SHALL calculate spike_rate as (current_volume / baseline)
2. WHEN spike_rate exceeds the significance threshold, THE Spike_Analyzer SHALL set is_significant to true
3. THE Spike_Analyzer SHALL classify spike_type as "organic", "media_driven", or "coordinated"
4. THE Spike_Analyzer SHALL classify spike_nature as "positive", "negative", "mixed", or "neutral"
5. THE Spike_Analyzer SHALL calculate actionability_score using the formula: (spike_intensity × 0.5) + (keyword_weight × 0.5)
6. WHEN trend_data contains is_partial flags, THE Spike_Analyzer SHALL set data_completeness to "confirmed", "partial", or "mixed"
7. WHEN data_completeness is "partial" or "mixed", THE Spike_Analyzer SHALL include a partial_data_warning message
8. THE Spike_Analyzer SHALL detect viral indicators including is_trending, has_breakout, and cross_platform presence
9. THE Spike_Analyzer SHALL calculate confidence score using: base_confidence × data_completeness_factor × spike_clarity_factor
10. THE Spike_Analyzer SHALL output a SpikeAnalysisResult with all required fields

### Requirement 3: Router Decision Logic - First Stage

**User Story:** As a system architect, I want to filter out insignificant spikes early, so that the system conserves resources and reduces noise.

#### Acceptance Criteria

1. WHEN SpikeAnalysisResult has is_significant equal to false, THE Router SHALL set route1_decision to "skip"
2. WHEN route1_decision is "skip", THE System SHALL log the event with trace_id, keyword, spike_rate, and reason
3. WHEN route1_decision is "skip", THE System SHALL set skipped to true and skip_reason to "not_significant"
4. WHEN route1_decision is "skip", THE System SHALL terminate the workflow without generating an Executive_Brief
5. WHEN SpikeAnalysisResult has is_significant equal to true, THE Router SHALL set route1_decision to "analyze"
6. WHEN route1_decision is "analyze", THE System SHALL proceed to Sentiment_Agent

### Requirement 4: Sentiment Analysis

**User Story:** As an analyst, I want to understand the emotional tone of social media messages, so that I can assess public sentiment toward the artist or issue.

#### Acceptance Criteria

1. WHEN messages are provided, THE Sentiment_Agent SHALL classify each message into sentiment categories: "support", "disappointment", "boycott", "meme_positive", "meme_negative", "fanwar", "neutral"
2. THE Sentiment_Agent SHALL calculate sentiment_distribution as proportions summing to 1.0
3. THE Sentiment_Agent SHALL identify dominant_sentiment as the category with highest proportion
4. WHEN a second category exceeds a threshold, THE Sentiment_Agent SHALL set secondary_sentiment
5. WHEN sentiment_distribution shows multiple significant categories, THE Sentiment_Agent SHALL set has_mixed_sentiment to true
6. THE Sentiment_Agent SHALL detect sentiment_shift as "worsening", "improving", "stable", or null
7. THE Sentiment_Agent SHALL extract representative_messages for each sentiment category
8. THE Sentiment_Agent SHALL identify meme_keywords when meme sentiments are detected
9. WHEN fanwar sentiment is detected, THE Sentiment_Agent SHALL identify fanwar_targets
10. THE Sentiment_Agent SHALL match messages against a lexicon and populate lexicon_matches with counts and types
11. THE Sentiment_Agent SHALL calculate confidence as: model_confidence × sample_quality_factor
12. THE Sentiment_Agent SHALL record analyzed_count as the number of messages processed

### Requirement 5: Router Decision Logic - Second Stage

**User Story:** As a system architect, I want to route simple fan reactions directly to playbook generation, so that the system avoids expensive analysis for low-priority cases.

#### Acceptance Criteria

1. WHEN actionability_score is less than 0.3, THE Router SHALL set route2_decision to "sentiment_only"
2. WHEN actionability_score is greater than or equal to 0.6, THE Router SHALL set route2_decision to "full_analysis"
3. WHEN actionability_score is between 0.3 and 0.6 AND boycott sentiment is greater than or equal to 0.2, THE Router SHALL set route2_decision to "full_analysis"
4. WHEN actionability_score is between 0.3 and 0.6 AND fanwar sentiment is greater than 0.1, THE Router SHALL set route2_decision to "full_analysis"
5. WHEN actionability_score is between 0.3 and 0.6 AND dominant_sentiment equals "meme_negative", THE Router SHALL set route2_decision to "full_analysis"
6. WHEN actionability_score is between 0.3 and 0.6 AND has_mixed_sentiment is true AND sentiment_shift equals "worsening", THE Router SHALL set route2_decision to "full_analysis"
7. WHEN actionability_score is between 0.3 and 0.6 AND spike_nature equals "positive" AND spike_rate is greater than or equal to 3.0 AND support sentiment is greater than or equal to 0.5, THE Router SHALL set positive_viral_detected to true and route2_decision to "full_analysis"
8. WHEN actionability_score is between 0.3 and 0.6 AND no crisis or opportunity signals are detected, THE Router SHALL set route2_decision to "sentiment_only"
9. WHEN route2_decision is "sentiment_only", THE System SHALL proceed directly to Playbook_Agent
10. WHEN route2_decision is "full_analysis", THE System SHALL proceed to Causality_Agent

### Requirement 6: Causality and Network Analysis

**User Story:** As an analyst, I want to identify influential accounts and propagation patterns, so that I can understand how information spreads and who drives the conversation.

#### Acceptance Criteria

1. WHEN spike_event messages are provided, THE Causality_Agent SHALL analyze the retweet network
2. THE Causality_Agent SHALL classify trigger_source as "influencer", "media", "organic", or "unknown"
3. THE Causality_Agent SHALL identify hub_accounts with influence_score, follower_count, and account_type
4. THE Causality_Agent SHALL calculate retweet_network_metrics including centralization and avg_degree
5. THE Causality_Agent SHALL classify cascade_pattern as "viral", "echo_chamber", or "coordinated"
6. WHEN origin can be determined, THE Causality_Agent SHALL set estimated_origin_time
7. THE Causality_Agent SHALL generate key_propagation_paths describing information flow

### Requirement 7: Router Decision Logic - Third Stage

**User Story:** As a system architect, I want to route positive viral opportunities to amplification analysis and potential crises to legal risk assessment, so that appropriate specialized analysis is performed.

#### Acceptance Criteria

1. WHEN positive_viral_detected is true, THE Router SHALL set route3_decision to "amplification"
2. WHEN positive_viral_detected is false or null, THE Router SHALL set route3_decision to "legal"
3. WHEN route3_decision is "amplification", THE System SHALL proceed to Amplification node
4. WHEN route3_decision is "legal", THE System SHALL proceed to Legal_RAG_Agent

### Requirement 8: Legal Risk Assessment

**User Story:** As a legal advisor, I want to identify potential legal risks in social media content, so that the company can respond appropriately to threats.

#### Acceptance Criteria

1. THE Legal_RAG_Agent SHALL perform a lightweight keyword check on all messages
2. THE Legal_RAG_Agent SHALL check for legal keywords: "고소", "소송", "법적", "재판", "표절", "저작권", "도용", "무단사용", "명예훼손", "허위사실", "딥페이크", "사생활", "유출", "폭로", "전속계약", "탈퇴", "역바이럴"
3. WHEN no legal keywords are detected AND spike_nature is "positive", THE Legal_RAG_Agent SHALL set clearance_status to "clear" and skip RAG retrieval
4. WHEN legal keywords are detected OR spike_nature is "negative" or "mixed", THE Legal_RAG_Agent SHALL perform RAG retrieval
5. WHEN RAG retrieval is performed, THE Legal_RAG_Agent SHALL search legal documents for relevant laws, clauses, and precedents
6. THE Legal_RAG_Agent SHALL set overall_risk_level to "low", "medium", "high", or "critical"
7. THE Legal_RAG_Agent SHALL set clearance_status to "clear", "review_needed", or "high_risk"
8. WHEN RAG is not performed, THE Legal_RAG_Agent SHALL set confidence to 0.95
9. WHEN RAG is performed, THE Legal_RAG_Agent SHALL calculate confidence as: rag_confidence × retrieval_quality_factor
10. THE Legal_RAG_Agent SHALL set rag_required and rag_performed flags appropriately
11. WHEN RAG is performed, THE Legal_RAG_Agent SHALL populate risk_assessment with risk_level, legal_violation, and analysis
12. WHEN RAG is performed, THE Legal_RAG_Agent SHALL provide recommended_action and referenced_documents

### Requirement 9: Amplification Opportunity Analysis

**User Story:** As a marketing strategist, I want to identify and leverage positive viral opportunities, so that the company can maximize beneficial exposure.

#### Acceptance Criteria

1. WHEN positive viral conditions are met, THE Amplification node SHALL extract top_platforms from cross_platform data
2. THE Amplification node SHALL identify hub_accounts with high influence_score
3. THE Amplification node SHALL extract representative_messages with high engagement
4. THE Amplification node SHALL generate suggested_actions for viral amplification
5. THE Amplification node SHALL output amplification_summary with platforms, accounts, messages, and actions

### Requirement 10: Strategic Playbook Generation

**User Story:** As a decision-maker, I want actionable strategic recommendations, so that I can respond effectively to crises or opportunities.

#### Acceptance Criteria

1. THE Playbook_Agent SHALL classify situation_type as "crisis", "opportunity", "monitoring", or "amplification"
2. THE Playbook_Agent SHALL set priority to "urgent", "high", "medium", or "low"
3. THE Playbook_Agent SHALL generate recommended_actions with action type, urgency, and description
4. WHEN action is "issue_statement" or "legal_response", THE Playbook_Agent SHALL include a draft field
5. WHEN action is "amplify_viral", THE Playbook_Agent SHALL include target_posts with id, source, source_message_id, and url
6. WHEN action is "legal_response", THE Playbook_Agent SHALL include legal_basis
7. THE Playbook_Agent SHALL identify key_risks based on analysis results
8. WHEN positive opportunities exist, THE Playbook_Agent SHALL identify key_opportunities
9. THE Playbook_Agent SHALL specify target_channels for communication

### Requirement 11: Executive Brief Generation

**User Story:** As an executive, I want a concise summary of the situation and recommended actions, so that I can make informed decisions quickly.

#### Acceptance Criteria

1. THE System SHALL generate an Executive_Brief with a summary statement
2. THE System SHALL calculate severity_score from 1 to 10 based on issue polarity and intensity
3. THE System SHALL set trend_direction to "escalating", "stable", or "declining"
4. THE System SHALL set issue_polarity to "positive", "negative", or "mixed"
5. THE System SHALL generate spike_summary from SpikeAnalysisResult
6. THE System SHALL generate sentiment_summary from SentimentAnalysisResult
7. THE System SHALL generate legal_summary from LegalRiskResult
8. THE System SHALL generate action_summary from PlaybookResult
9. WHEN issue_polarity is "positive", THE System SHALL generate opportunity_summary
10. THE System SHALL populate analysis_status for each agent: "success", "failed", or "skipped"
11. WHEN any agent fails, THE System SHALL set user_message to inform about limited analysis
12. THE System SHALL record generated_at timestamp and analysis_duration_seconds

### Requirement 12: Error Handling and Resilience

**User Story:** As a system operator, I want the system to handle failures gracefully, so that partial results are still delivered when components fail.

#### Acceptance Criteria

1. WHEN an agent encounters an error, THE System SHALL add an entry to error_logs with stage, error_type, message, occurred_at, and trace_id
2. WHEN an agent fails, THE System SHALL continue workflow execution with remaining agents
3. THE System SHALL use the same trace_id for all operations in a workflow
4. WHEN an agent fails, THE System SHALL set the corresponding analysis_status field to "failed"
5. WHEN partial results are available, THE System SHALL generate an Executive_Brief with available information
6. THE System SHALL set user_message to explain limitations when failures occur
7. THE System SHALL log errors using a structured logger for debugging

### Requirement 13: State Management and Traceability

**User Story:** As a system architect, I want comprehensive state tracking throughout the workflow, so that I can debug issues and audit decisions.

#### Acceptance Criteria

1. THE System SHALL maintain an AnalysisState object throughout the workflow
2. THE System SHALL populate route1_decision, route2_decision, and route3_decision at each router stage
3. THE System SHALL store results from each agent in the corresponding state field
4. THE System SHALL populate node_insights with a one-line summary from each agent
5. THE System SHALL preserve trace_id from Kafka message through entire workflow
6. THE System SHALL record workflow_start_time at workflow initiation
7. WHEN workflow is skipped, THE System SHALL set skipped to true and populate skip_reason

### Requirement 14: Data Format and Schema Compliance

**User Story:** As a developer, I want all data structures to follow defined schemas, so that integration between components is reliable.

#### Acceptance Criteria

1. THE System SHALL use UUID format for message_id and id fields
2. THE System SHALL use ISO 8601 format for all timestamp fields
3. THE System SHALL use source enum values: "twitter", "theqoo", "instiz", "google_trends"
4. THE System SHALL use lowercase with underscores for internal enum values
5. THE System SHALL preserve external API enum values in original format
6. THE System SHALL ensure all confidence fields are in range 0.0 to 1.0
7. THE System SHALL ensure all proportion fields sum to 1.0 where applicable
8. THE System SHALL use TypedDict definitions for all structured data

### Requirement 15: Performance and Cost Optimization

**User Story:** As a system operator, I want the system to minimize costs while maintaining quality, so that the service is economically sustainable.

#### Acceptance Criteria

1. WHEN actionability_score is below 0.3, THE System SHALL skip expensive Causality and Legal analysis
2. WHEN legal keywords are not detected, THE Legal_RAG_Agent SHALL skip RAG retrieval
3. THE System SHALL complete lightweight keyword checks within 0.1 seconds
4. WHEN RAG retrieval is required, THE System SHALL complete within 12 seconds
5. THE System SHALL log skip decisions for cost analysis
6. THE System SHALL track analysis_duration_seconds for performance monitoring

## Non-Functional Requirements

### Performance

- The system SHALL process a spike event end-to-end within 30 seconds for full_analysis path
- The system SHALL process a spike event end-to-end within 5 seconds for sentiment_only path
- The system SHALL handle at least 100 concurrent spike events

### Reliability

- The system SHALL maintain 99% uptime during business hours
- The system SHALL recover from individual agent failures without workflow termination

### Scalability

- The system SHALL support adding new data sources without modifying core workflow
- The system SHALL support adding new sentiment categories without breaking existing analysis

### Security

- The system SHALL anonymize personal information in messages when is_anonymized flag is set
- The system SHALL not expose raw message content in logs

### Maintainability

- All components SHALL use TypedDict for type safety
- All components SHALL follow the defined function interfaces
- All components SHALL log errors with structured trace_id for debugging
