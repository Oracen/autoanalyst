# Autoanalyst

**Automated Business Analytics Framework**

Autoanalyst transforms routine business analytics by automatically identifying root causes of KPI changes, freeing analysts from repetitive "why did X happen?" queries to focus on real-world impact analysis.

## Overview

Traditional business analytics requires analysts to manually decompose metrics, identify significant drivers, and aggregate similar dimensional breakdowns. Autoanalyst automates this entire pipeline through:

1. **Metric Tree Decomposition**: Model KPI relationships with strict or flexible definitions - everything rolls up and converts to top-level units
2. **Signature-Based Clustering**: Collapse similar-behaving dimensions (e.g., "these 5 small cities act identically") using path signatures to map arbitrary time series into fixed-length vectors
3. **Automated Driver Identification**: Surface major movers and component-level changes in native KPI units
4. **Natural Language Input**: Generate reports and address follow-up questions with a Slack-ready chatbot! Amaze friends and coworkers by responding to the CEO while AFK!

## Target Use Case

Tech-heavy, time-strapped analytics teams who need to offload routine investigative work. Instead of analysts manually drilling down through dimensional hierarchies, the system pre-computes, pre-prioritizes, and pre-explains the key drivers of business changes.

## Current Status

🟢 **Complete**: Metric tree decomposition and signature-based time series clustering  
🟡 **In Development**: Statistical process control ruleset, AI agent integration  
🔴 **Planned**: Natural language report generation and chat interface  

⚠️ **Work in Progress**: This is experimental software. Dependencies and APIs will change.
