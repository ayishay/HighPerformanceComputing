# GenAI Risk Management: Scalable Guardrail Strategy for Banking — Complete Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Core Problem](#2-the-core-problem)
3. [What Are Guardrails?](#3-what-are-guardrails)
4. [The Solution: Three-Tier Guardrail Model](#4-the-solution-three-tier-guardrail-model)
5. [Guardrail Template Library](#5-guardrail-template-library)
6. [Tier 3 Self-Service Portal: A No-Code Guide for Business Teams](#6-tier-3-self-service-portal-a-no-code-guide-for-business-teams)
7. [Risk Assessment Framework](#7-risk-assessment-framework)
8. [Governance Process](#8-governance-process)
9. [Risk-Tiered Oversight](#9-risk-tiered-oversight)
10. [Continuous Improvement Loop](#10-continuous-improvement-loop)
11. [Roles and Responsibilities](#11-roles-and-responsibilities)
12. [Summary](#12-summary)

---

## 1. Introduction

### Who Is This Guide For?

This guide is for risk managers, compliance professionals, technology leaders, and anyone involved in deploying or overseeing Generative AI (GenAI) — particularly Large Language Models (LLMs) — within a banking or financial services organization.

### What Will You Learn?

By the end of this guide, you will understand:

- Why traditional centralized guardrail creation does not scale
- How to build a tiered guardrail framework that scales to thousands of use cases
- How to assess and classify the risk of each GenAI use case
- How to govern GenAI deployments without creating bottlenecks
- How to monitor and continuously improve your guardrail framework

### Why Does This Matter?

Banks are rapidly adopting GenAI for tasks like document summarization, customer service, code generation, compliance research, and more. Each of these use cases carries unique risks — from leaking customer data to generating biased lending recommendations. Without proper guardrails, a single failure could result in regulatory penalties, reputational damage, financial loss, or harm to customers.

---

## 2. The Core Problem

### The Bottleneck

Most banks follow this pattern when deploying GenAI:

1. The **technology department** provides an LLM platform (e.g., a chatbot, an API, or an internal tool)
2. They add a few **general guardrails** — basic content filters, PII detection, etc.
3. Business teams across the bank want to use the LLM for **thousands of different use cases**
4. Each use case has **unique risks** that require unique guardrails

**The problem:** It is impossible for one centralized tech team to design, build, and maintain individual guardrails for every single use case. There are too many use cases, and the tech team doesn't have the domain expertise to understand the specific risks of each one.

### What Happens Without a Scalable Solution?

| Scenario | Consequence |
|----------|-------------|
| Tech team tries to build everything | Massive backlog; use cases wait months to deploy |
| Business teams deploy without guardrails | Uncontrolled risk; regulatory exposure |
| One-size-fits-all guardrails | Too restrictive for some use cases, too permissive for others |
| Ad-hoc guardrail requests | Inconsistent controls; no enterprise-wide standards |

### The Goal

Build a system where:

- **Universal protections** are always enforced automatically
- **Domain-specific rules** are standardized per business line
- **Use-case-specific controls** can be configured by the teams who know their use case best
- **Risk oversight** is proportional to actual risk — not a one-size-fits-all burden

---

## 3. What Are Guardrails?

### Definition

**Guardrails** are safety controls placed around a GenAI model to prevent it from producing harmful, inaccurate, non-compliant, or inappropriate outputs. Think of them like the bumper rails in a bowling alley — they keep the ball (the model's output) in the lane (acceptable behavior).

### Types of Guardrails

Guardrails can be applied at different stages of the interaction:

```
User sends a prompt
        │
        ▼
┌─────────────────┐
│  INPUT CONTROLS  │  ← Check what goes INTO the model
│  (Pre-Processing)│     - Is the prompt appropriate?
│                  │     - Does it contain sensitive data?
│                  │     - Is it within the allowed topic scope?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SYSTEM CONTROLS │  ← Control HOW the model behaves
│  (Context)       │     - What role should the model play?
│                  │     - What data sources can it use?
│                  │     - What disclaimers must it include?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OUTPUT CONTROLS │  ← Check what comes OUT of the model
│  (Post-Process)  │     - Is the output factually grounded?
│                  │     - Does it contain PII?
│                  │     - Does it violate regulations?
│                  │     - Is it biased or toxic?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OPERATIONAL     │  ← Monitor and manage the system
│  CONTROLS        │     - Logging and audit trails
│                  │     - Human review when needed
│                  │     - Kill switches for emergencies
└─────────────────┘
```

### Guardrails in Banking Context — Examples

| Risk | Example Scenario | Guardrail |
|------|-----------------|-----------|
| **Data leakage** | User pastes customer SSN into the prompt | PII detection and redaction before the prompt reaches the model |
| **Unauthorized decisions** | Model says "Your loan is approved" | Decision boundary enforcer blocks the model from making credit decisions |
| **Regulatory violation** | Model uses discriminatory language in a lending context | Fair lending language filter screens output |
| **Hallucination** | Model invents a policy that doesn't exist | Factual grounding check validates output against source documents |
| **Scope creep** | User asks the credit memo tool to write poetry | Topic scope restrictor limits the model to its intended purpose |
| **Prompt injection** | Malicious user tries to trick the model into ignoring its rules | Prompt injection detector identifies and blocks the attack |

---

## 4. The Solution: Three-Tier Guardrail Model

The scalable solution is a **three-tier, federated guardrail framework**. Each tier has a different owner, scope, and governance model.

### Overview

```
┌──────────────────────────────────────────────────────────┐
│                    TIER 1: UNIVERSAL                      │
│              (Owned by Technology / Platform)              │
│                                                           │
│   Applied to ALL use cases. Cannot be disabled.           │
│   Examples: PII detection, prompt injection defense,      │
│   content toxicity filters, audit logging                 │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │              TIER 2: DOMAIN-LEVEL                 │    │
│  │     (Owned by Business Line Risk + 2nd Line)      │    │
│  │                                                    │    │
│  │   Applied per business domain.                     │    │
│  │   Examples: Lending cannot generate credit          │    │
│  │   decisions; Trading must include disclaimers       │    │
│  │                                                    │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │        TIER 3: USE-CASE-SPECIFIC          │     │    │
│  │  │     (Owned by Use Case Owners / 1st Line)  │     │    │
│  │  │                                            │     │    │
│  │  │   Configured per individual use case.       │     │    │
│  │  │   Examples: Topic restrictions, output       │     │    │
│  │  │   format rules, grounding thresholds         │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Tier 1: Universal Guardrails

**Owner:** Technology / Platform Team

**Scope:** Every single GenAI use case across the entire bank

**Key Principle:** These are non-negotiable. No use case owner can disable or weaken them.

**What belongs in Tier 1:**

| Control | Why It's Universal |
|---------|-------------------|
| PII / Sensitive Data Detection & Redaction | Every use case handles data; PII leakage is always a risk |
| Prompt Injection Defense | Every use case that accepts user input is vulnerable |
| Content Toxicity Filters | No use case should produce hate speech, violence, or explicit content |
| Audit Logging | Regulatory requirement; needed for every interaction |
| Data Leakage Prevention | Prevents model from exposing training data or internal system information |
| Basic Output Safety | Prevents the model from providing instructions for illegal activities |

**Who manages it:** The centralized technology/platform team builds and maintains these. They are embedded in the platform itself, not configured per use case.

**How it's enforced:** Hardcoded into the platform. Use case owners interact with the model through an API or interface that already has Tier 1 guardrails active. They cannot bypass them.

### Tier 2: Domain-Level Guardrails

**Owner:** Business Line Risk Leads + 2nd Line of Defense (Risk Management)

**Scope:** All use cases within a specific business domain (e.g., all lending use cases, all trading use cases, all AML/KYC use cases)

**Key Principle:** These reflect the regulatory and business risk profile of an entire domain. Use case owners within the domain can tighten them but cannot loosen them.

**Examples by domain:**

| Domain | Tier 2 Guardrail | Rationale |
|--------|------------------|-----------|
| **Lending** | Block any output that implies credit approval/denial | Fair Lending, ECOA compliance |
| **Lending** | Filter for discriminatory language (race, age, gender, national origin) | ECOA, Fair Housing Act |
| **Trading** | Require disclaimer: "Not investment advice" | SEC/FINRA requirements |
| **Trading** | Block specific securities recommendations | Suitability/fiduciary obligations |
| **AML/KYC** | Block output of SAR-related details to non-authorized users | BSA/AML confidentiality |
| **Customer Service** | Require empathetic, non-dismissive tone | UDAAP, customer experience standards |
| **Wealth Management** | Block guaranteed return language | SEC advertising rules |
| **Retail Banking** | Block fee waiver promises | Operational risk, P&L impact |

**Who creates them:** Domain risk leads, in collaboration with your 2nd line risk team, using templates and guidance provided by the platform team.

**How it's enforced:** When a use case is registered under a domain (e.g., "Lending"), the Tier 2 guardrails for that domain are automatically applied. The use case owner can add more restrictions on top but cannot remove domain-level controls.

### Tier 3: Use-Case-Specific Guardrails

**Owner:** Use Case Owner (the business team deploying the specific use case)

**Scope:** One individual use case only

**Key Principle:** These are configured by the people who understand their use case best. They use a self-service toolkit of pre-built, composable guardrail modules — they configure, not code.

**Examples:**

| Use Case | Tier 3 Guardrail | Why |
|----------|------------------|-----|
| Credit Memo Summarizer | Restrict topic to credit memo summarization only | Prevent misuse as a general-purpose chatbot |
| Credit Memo Summarizer | Require 85% grounding score against source documents | Ensure summaries are factually based on the actual memo |
| Internal Policy Q&A Bot | Limit knowledge to the HR policy corpus only | Prevent hallucinated policy answers |
| Internal Policy Q&A Bot | Require citation of source document and section | Enable verification |
| Customer Email Drafter | Enforce professional tone and maximum 200-word output | Brand consistency |
| Code Review Assistant | Block output containing production credentials or API keys | Security |

**Who creates them:** The use case owner, using the Guardrail Template Library (described in Section 5).

**How it's enforced:** The use case owner selects and configures modules from the library. The system validates that their configuration meets the minimum requirements for their risk tier and domain before deployment.

### Why This Model Works

| Challenge | How the Three-Tier Model Solves It |
|-----------|-----------------------------------|
| Tech team can't build thousands of guardrails | They only build Tier 1. Tier 2 and 3 are delegated. |
| Business teams lack technical skills | They use a self-service library of pre-built modules — configuration, not coding. |
| Need consistent enterprise standards | Tier 1 guarantees a universal baseline. Tier 2 guarantees domain consistency. |
| Need flexibility for diverse use cases | Tier 3 allows customization within the safety boundaries set by Tier 1 and 2. |
| Risk team can't review everything | Risk-tiered oversight focuses attention on the highest-risk use cases. |

---

## 5. Guardrail Template Library

### What Is It?

The Guardrail Template Library is a **catalog of pre-built, reusable guardrail modules** that use case owners can select and configure — like choosing toppings for a pizza. Each module addresses a specific type of risk and has configurable parameters.

Think of it as an internal "app store" for risk controls. Use case owners browse the catalog, select the modules relevant to their use case, configure the parameters, and deploy.

### Architecture

The library has three layers:

```
┌─────────────────────────────────────────────────────┐
│        GUARDRAIL CONFIGURATION LAYER                 │
│                                                      │
│   This is what use case owners interact with.        │
│   A form, portal, or YAML config file where they     │
│   select modules and set parameters.                 │
├─────────────────────────────────────────────────────┤
│        GUARDRAIL ENGINE LAYER                        │
│                                                      │
│   This orchestrates the modules at runtime.           │
│   It decides the execution order, handles conflicts, │
│   and ensures all checks run before output is         │
│   delivered to the user.                              │
├─────────────────────────────────────────────────────┤
│        GUARDRAIL MODULE LIBRARY                      │
│                                                      │
│   The actual collection of individual guardrail       │
│   components. Each module is a self-contained,       │
│   reusable control.                                  │
└─────────────────────────────────────────────────────┘
```

### Module Catalog

Below is the complete catalog of guardrail modules, organized by where they act in the interaction flow.

#### Category 1: Input Controls (Pre-Processing)

These modules check and filter what goes INTO the model before it processes anything.

| Module ID | Module Name | What It Does | Configurable Parameters |
|-----------|-------------|--------------|------------------------|
| IN-01 | **Topic Scope Restrictor** | Rejects or redirects prompts that fall outside the defined topic boundaries for this use case. Prevents the tool from being used for unintended purposes. | Allowed topic list, blocked topic list, similarity threshold (how closely a prompt must match a topic to be accepted), custom rejection message |
| IN-02 | **PII / Sensitive Data Scanner** | Scans the user's prompt for personally identifiable information (SSN, account numbers, dates of birth, etc.) and takes action before the data reaches the model. | Which data types to scan for, action per data type (redact the data and continue / block the entire prompt / show a warning), redaction format (e.g., replace SSN with `***-**-****`) |
| IN-03 | **Prompt Injection Detector** | Identifies attempts by users to trick the model into ignoring its instructions (known as "jailbreaking" or "prompt injection"). | Detection sensitivity (low / medium / high), action (block the prompt / flag for review / log only), who to escalate to if flagged |
| IN-04 | **User Authentication & Entitlement Check** | Verifies that the user is authorized to use this specific use case and access the data it uses. | Required roles or permissions, data entitlements (e.g., can only access data for their region), geographic restrictions |
| IN-05 | **Input Length & Complexity Limiter** | Prevents abuse through excessively long or complex prompts that could cause performance issues or exploit the model. | Maximum token (word) count, maximum file attachment size, which file types are allowed |
| IN-06 | **Regulatory Keyword Flagger** | Detects prompts containing words or phrases that have regulatory significance and may require special handling. | Keyword/phrase lists organized by regulation (UDAAP, BSA, Fair Lending, etc.), action per regulation (block / flag for review / log only) |

#### Category 2: System Prompt & Context Controls

These modules control the model's behavior by shaping its instructions and available information.

| Module ID | Module Name | What It Does | Configurable Parameters |
|-----------|-------------|--------------|------------------------|
| SYS-01 | **Persona & Role Enforcer** | Defines who the model "is" — its role, boundaries, and communication style. This is injected as a system prompt that the model follows. | Persona template (e.g., "You are a credit memo summarization assistant"), prohibited behaviors (e.g., "Never provide investment advice"), required tone (e.g., professional, formal) |
| SYS-02 | **Knowledge Boundary Setter** | Restricts the model to only use specific, approved data sources when generating responses (critical for RAG — Retrieval-Augmented Generation — use cases). | List of allowed data source IDs, what to do when no relevant source is found (say "I don't know" vs. attempt an answer), whether citations are required in the output |
| SYS-03 | **Disclaimer Injector** | Ensures that required legal or compliance disclaimers are always included in the model's output. | Disclaimer text per domain, where to place it (beginning or end of response), conditions that trigger the disclaimer |
| SYS-04 | **Conversation Memory Policy** | Controls how the model handles conversation history — how much it remembers and how it protects sensitive information across turns. | Maximum number of conversation turns to retain, whether to scrub PII from previous turns, session timeout (auto-clear after inactivity) |

#### Category 3: Output Controls (Post-Processing)

These modules check and filter what comes OUT of the model before the user sees it.

| Module ID | Module Name | What It Does | Configurable Parameters |
|-----------|-------------|--------------|------------------------|
| OUT-01 | **Output PII Scanner** | Scans the model's response for PII that shouldn't be exposed — the model may generate PII even if none was in the prompt (from training data or context). | Same configuration as IN-02, but applied to the output side |
| OUT-02 | **Factual Grounding Checker** | For RAG use cases, validates that the claims in the model's output are actually supported by the source documents provided. Prevents hallucination. | Grounding threshold (e.g., 85% of claims must be sourced), action when threshold is not met (block / flag for human review / add warning), citation format requirements |
| OUT-03 | **Toxicity & Bias Filter** | Screens the model's output for harmful, biased, discriminatory, or inappropriate content. | Sensitivity level (low / medium / high), protected categories to monitor (race, gender, age, disability, etc.), action (block / rephrase / flag for review) |
| OUT-04 | **Confidence & Certainty Qualifier** | Ensures the model doesn't present uncertain information as fact. Adds hedging language or qualifiers when the model's confidence is low. | Required hedging phrases (e.g., "Based on the available information..."), confidence score threshold below which qualifiers are mandatory, qualifier text template |
| OUT-05 | **Regulatory Compliance Checker** | Validates the model's output against domain-specific regulatory language rules. | Which regulation rulesets to check against (Fair Lending, UDAAP, advertising rules, etc.), action when a violation is detected (block / flag / rephrase) |
| OUT-06 | **Output Format Enforcer** | Ensures the model's output conforms to a required structure or format. | Allowed formats (plain text / JSON / table / bullet points), maximum output length, required sections in the output, prohibited content patterns |
| OUT-07 | **Decision Boundary Enforcer** | Prevents the model from making, implying, or appearing to make decisions that require human judgment. Critical in banking where many decisions have legal implications. | List of prohibited action verbs/phrases (e.g., "approved", "denied", "eligible", "recommended"), escalation message to display instead (e.g., "Credit decisions must be made by an authorized credit officer") |

#### Category 4: Operational Controls

These modules manage the ongoing operation, monitoring, and governance of the use case.

| Module ID | Module Name | What It Does | Configurable Parameters |
|-----------|-------------|--------------|------------------------|
| OPS-01 | **Audit Logger** | Records all interactions (prompts and responses) for compliance, investigation, and monitoring purposes. | Log detail level (full prompt + response / metadata only), retention period, storage destination, whether to include guardrail trigger events |
| OPS-02 | **Human-in-the-Loop Gate** | Routes the model's output to a human reviewer for approval before delivering it to the user. Used for high-risk outputs. | Trigger conditions (always require review / only when confidence is below threshold / only when a guardrail was triggered), reviewer role, maximum review time before escalation |
| OPS-03 | **Rate Limiter & Abuse Detector** | Prevents excessive or suspicious usage patterns that could indicate abuse, data exfiltration attempts, or system misuse. | Maximum requests per user per time window, anomaly detection sensitivity, what happens when limits are exceeded (temporary lockout / alert to security / require manager approval) |
| OPS-04 | **Feedback Collector** | Captures user feedback on the quality and accuracy of the model's output. This data is essential for monitoring and improvement. | Feedback type (thumbs up/down, free text comments, accuracy rating), whether feedback is mandatory or optional, where feedback data is stored |
| OPS-05 | **Kill Switch** | Enables immediate shutdown of a use case if a critical issue is discovered. Every use case should have one. | Trigger mechanism (manual button / automated when error rate exceeds threshold), who gets notified, fallback message to show users when the use case is disabled |
| OPS-06 | **Model Drift Monitor** | Tracks output quality metrics over time and alerts when the model's behavior changes in unexpected ways. Models can degrade or drift — this catches it early. | Which metrics to track (refusal rate, average output length, sentiment distribution, guardrail trigger frequency), acceptable threshold ranges, where to send alerts |

### How Use Case Owners Assemble Guardrails

A use case owner configures their guardrails by filling out a structured form or configuration file. Here is an example for a "Commercial Lending — Credit Memo Summarizer" use case:

```yaml
use_case:
  id: UC-2024-0387
  name: "Commercial Lending - Credit Memo Summarizer"
  owner: "J. Smith, Commercial Lending"
  risk_tier: HIGH
  business_line: Commercial Banking

guardrails:

  # ──────────────────────────────────────────────
  # TIER 1 — Automatically applied. Cannot remove.
  # IN-02 (PII Scanner), IN-03 (Prompt Injection),
  # OUT-01 (Output PII), OPS-01 (Audit Logging)
  # are always active for every use case.
  # ──────────────────────────────────────────────

  # ──────────────────────────────────────────────
  # TIER 2 — Domain-required for Lending
  # Automatically applied because business_line = Commercial Banking
  # ──────────────────────────────────────────────
  - module: OUT-05  # Regulatory Compliance Checker
    config:
      regulation_ruleset:
        - "fair_lending"
        - "udaap"
        - "ecoa"
      action: block_and_escalate

  - module: OUT-07  # Decision Boundary Enforcer
    config:
      prohibited_phrases:
        - "approved"
        - "denied"
        - "eligible"
        - "qualified"
      escalation_message: >
        This tool provides summaries only.
        All credit decisions must be made by authorized personnel.

  # ──────────────────────────────────────────────
  # TIER 3 — Configured by the use case owner
  # ──────────────────────────────────────────────
  - module: IN-01  # Topic Scope Restrictor
    config:
      allowed_topics:
        - "credit_memo_summarization"
        - "financial_statement_analysis"
      blocked_topics:
        - "credit_decisioning"
        - "pricing"
        - "personal_advice"
      rejection_message: >
        This tool only summarizes credit memos.
        Please contact your credit officer for other requests.

  - module: SYS-02  # Knowledge Boundary Setter
    config:
      allowed_data_sources:
        - "credit_memo_corpus"
        - "financial_policy_docs"
      citation_required: true

  - module: OUT-02  # Factual Grounding Checker
    config:
      grounding_threshold: 0.85
      action: flag_for_review

  - module: OPS-02  # Human-in-the-Loop Gate
    config:
      trigger: "confidence_below_threshold"
      confidence_threshold: 0.70
      reviewer_role: "credit_analyst"

  - module: OUT-04  # Confidence Qualifier
    config:
      required_qualifier: >
        This summary is auto-generated and must be reviewed
        by a credit officer before use in any credit decision.
```

### Governance Rules for the Library

| Rule | What It Means |
|------|---------------|
| **Tier 1 modules are hardcoded** | Use case owners cannot disable them. They are always active. |
| **Tier 2 modules are domain-mandatory** | Automatically applied based on business line. Owners can make them stricter but cannot weaken or remove them. |
| **Tier 3 modules are owner-configured** | Selected and tuned by the use case owner from the library catalog. |
| **Minimum coverage required** | The system rejects configurations that don't meet the minimum guardrail requirements for the assessed risk tier. For example, a HIGH-risk use case must have a human-in-the-loop gate. |
| **Version control** | All guardrail configurations are version-controlled with full change history and approval trails. Every change is traceable. |

---

## 6. Tier 3 Self-Service Portal: A No-Code Guide for Business Teams

### The Problem This Section Solves

Tier 3 guardrails are owned by **use case owners** — and in most banks, these owners are **business people**, not engineers. They are loan officers, compliance analysts, marketing managers, operations leads. They understand their business deeply, but they do not write code, edit YAML files, or configure technical systems.

So how do they set up Tier 3 guardrails?

The answer: **they fill out forms.** The platform team builds a self-service web portal (think of it like an internal website) where business teams answer plain-English questions, pick options from dropdowns, and toggle switches. Behind the scenes, the portal automatically translates those form answers into the technical guardrail configuration. The business user never sees code, never touches a config file, and never needs to understand the underlying technology.

### How It Works: The Big Picture

```
WHAT THE BUSINESS USER SEES          WHAT HAPPENS BEHIND THE SCENES
================================     =====================================

Step 1: Open the portal website  →   Portal loads their use case info
                                      (name, domain, risk tier) from
                                      the registration in Step 1-4
                                      of the Governance Process

Step 2: Answer guided questions  →   Portal maps answers to guardrail
        via forms and dropdowns       module IDs and parameter values

Step 3: Review a plain-English   →   Portal generates the technical
        summary of their choices      YAML/JSON configuration file

Step 4: Click "Submit for        →   Configuration enters the review
        Review"                       and approval workflow (Section 8)

Step 5: Receive approval         →   Platform team deploys the
        notification                  configuration to production
```

The key insight: **the business user's experience is entirely form-based**. Everything technical is abstracted away.

### Architecture: How Forms Become Guardrails

Below is the full architecture of the self-service portal, from the business user's browser to the running guardrail in production.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│   LAYER 1: SELF-SERVICE WEB PORTAL (what business users see)     │
│                                                                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │  Step 1   │  │  Step 2   │  │  Step 3   │  │  Step 4   │       │
│   │  Use Case │→│  Guardrail│→│  Review & │→│  Submit   │       │
│   │  Info     │  │  Question-│  │  Confirm  │  │  for      │       │
│   │  (auto-   │  │  naire    │  │  (plain   │  │  Approval │       │
│   │  filled)  │  │  (forms)  │  │  English) │  │           │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   LAYER 2: TRANSLATION ENGINE (invisible to business users)      │
│                                                                   │
│   • Maps form answers → guardrail module IDs + parameters        │
│   • Applies business rules (e.g., HIGH risk → human review       │
│     is mandatory, cannot be skipped)                              │
│   • Merges with auto-applied Tier 1 and Tier 2 configs           │
│   • Validates: are all required guardrails present for this       │
│     risk tier? Flags gaps before submission.                      │
│   • Generates the technical config (YAML/JSON)                   │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   LAYER 3: REVIEW & APPROVAL WORKFLOW                            │
│                                                                   │
│   • Routes to the right approver based on risk tier               │
│   • LOW → auto-approved if validation passes                     │
│   • MEDIUM → domain risk lead reviews                            │
│   • HIGH → 2nd Line of Defense reviews                           │
│   • CRITICAL → risk committee reviews                            │
│   • Approver sees BOTH the plain-English summary AND the         │
│     generated technical config                                    │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   LAYER 4: DEPLOYMENT PIPELINE (handled by platform team)        │
│                                                                   │
│   • Approved config is version-controlled (audit trail)          │
│   • Deployed to the guardrail engine                              │
│   • Tier 1 + Tier 2 + Tier 3 configs merged at runtime           │
│   • Monitoring and alerting activated                             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### What the Forms Actually Look Like

Below is what the business user actually sees — step by step — when they configure Tier 3 guardrails through the portal. Each section of the form corresponds to a guardrail module from the Template Library (Section 5), but the user never needs to know module IDs or technical details.

#### Form Section 1: Use Case Identity (Auto-Filled)

This section is pre-populated from the use case registration. The business user just confirms it is correct.

```
┌─────────────────────────────────────────────────────────────┐
│  GUARDRAIL CONFIGURATION PORTAL                              │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  Use Case Name:     [Commercial Lending - Credit Memo       │
│                      Summarizer                        ]     │
│  Use Case ID:       UC-2024-0387                             │
│  Owner:             J. Smith, Commercial Lending             │
│  Business Domain:   Commercial Banking (auto-detected)       │
│  Risk Tier:         HIGH (calculated from risk assessment)   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ℹ  Because your use case is in "Commercial Banking" │    │
│  │    and is rated HIGH risk, the following guardrails  │    │
│  │    are ALREADY applied automatically:                │    │
│  │                                                      │    │
│  │    TIER 1 (Universal — always on):                   │    │
│  │    ✓ PII detection & redaction                       │    │
│  │    ✓ Prompt injection defense                        │    │
│  │    ✓ Content toxicity filter                         │    │
│  │    ✓ Audit logging                                   │    │
│  │                                                      │    │
│  │    TIER 2 (Commercial Banking domain — always on):   │    │
│  │    ✓ Fair lending compliance checker                  │    │
│  │    ✓ Credit decision boundary enforcer               │    │
│  │    ✓ Regulatory language filter (ECOA, UDAAP)        │    │
│  │                                                      │    │
│  │    You cannot remove these. You will now configure    │    │
│  │    additional guardrails specific to YOUR use case.   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  [ Next → ]                                                  │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 2: What Should This Tool Be Used For?

This maps to module **IN-01: Topic Scope Restrictor**. But the business user just sees a plain-English question.

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT SHOULD THIS TOOL BE USED FOR?                          │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  Describe what this tool is supposed to do (in plain         │
│  English). The system will restrict the tool to only         │
│  respond to requests related to these topics.                │
│                                                              │
│  What topics should this tool handle? (check all that apply) │
│                                                              │
│  ☑  Credit memo summarization                                │
│  ☑  Financial statement analysis                             │
│  ☐  Credit decisioning                                       │
│  ☐  Loan pricing                                             │
│  ☐  General Q&A / open-ended chat                            │
│  ☐  Customer communication drafting                          │
│  [+ Add custom topic: _________________________ ]            │
│                                                              │
│  What should this tool NOT do? (check all that apply)        │
│                                                              │
│  ☑  Make or imply credit decisions                           │
│  ☑  Provide pricing or rate quotes                           │
│  ☑  Give personal financial advice                           │
│  ☐  Draft customer-facing communications                     │
│  [+ Add custom restriction: ___________________ ]            │
│                                                              │
│  When someone asks an off-topic question, what message       │
│  should they see?                                            │
│                                                              │
│  [ This tool only summarizes credit memos. Please contact  ] │
│  [ your credit officer for other requests.                 ] │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 3: What Data Sources Can the Tool Use?

This maps to module **SYS-02: Knowledge Boundary Setter**.

```
┌─────────────────────────────────────────────────────────────┐
│  WHAT DATA SOURCES CAN THE TOOL USE?                         │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  Select the approved data sources this tool can reference    │
│  when generating responses. The tool will ONLY use these     │
│  sources — it will not make up information from other        │
│  sources.                                                    │
│                                                              │
│  Available data sources for your domain:                     │
│                                                              │
│  ☑  Credit memo corpus                                       │
│  ☑  Financial policy documents                               │
│  ☐  Customer account database                                │
│  ☐  Market data feeds                                        │
│  ☐  Internal procedures & guidelines                         │
│  ☐  Regulatory reference library                             │
│                                                              │
│  Must the tool cite its sources in every response?           │
│                                                              │
│  ● Yes — require a citation for every claim  (Recommended   │
│          for HIGH risk use cases)                             │
│  ○ No  — citations are optional                              │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 4: How Accurate Must the Responses Be?

This maps to module **OUT-02: Factual Grounding Checker**.

```
┌─────────────────────────────────────────────────────────────┐
│  HOW ACCURATE MUST THE RESPONSES BE?                         │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  This controls how strictly the tool checks that its         │
│  responses are supported by the source documents.            │
│                                                              │
│  Set the accuracy threshold:                                 │
│                                                              │
│  ○ Standard (70%) — Allows more flexible responses, but      │
│    some claims may not be directly traceable to sources.     │
│    Best for: brainstorming, exploratory analysis.            │
│                                                              │
│  ● Strict (85%) — Most claims must be directly supported     │
│    by source documents. Good balance of accuracy and         │
│    usefulness. (Recommended for HIGH risk)                   │
│                                                              │
│  ○ Very Strict (95%) — Nearly every claim must be directly   │
│    quoted or paraphrased from sources. May refuse to         │
│    answer if sources are insufficient.                       │
│    Best for: regulatory, legal, or audit-sensitive tasks.    │
│                                                              │
│  When a response falls below the accuracy threshold,         │
│  what should happen?                                         │
│                                                              │
│  ○ Block the response entirely                               │
│  ● Flag it for human review before delivering                │
│  ○ Deliver it with a warning label                           │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 5: Does a Human Need to Review Responses?

This maps to module **OPS-02: Human-in-the-Loop Gate**.

```
┌─────────────────────────────────────────────────────────────┐
│  DOES A HUMAN NEED TO REVIEW RESPONSES?                      │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ⚠  Because your use case is rated HIGH risk, at      │    │
│  │    least one human review trigger is REQUIRED.        │    │
│  │    You cannot skip this section.                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  When should a human review the tool's response before       │
│  the user sees it?                                           │
│                                                              │
│  ○ Always — every response is reviewed by a human            │
│  ● When confidence is low — only when the tool is less       │
│    than [ 70 ]% confident in its response                    │
│  ○ When a guardrail is triggered — only when the tool        │
│    flags a potential issue                                    │
│                                                              │
│  Who should review?                                          │
│                                                              │
│  Role:  [ Credit Analyst          ▼ ]                        │
│                                                              │
│  If no reviewer responds within [ 4 ] hours:                 │
│                                                              │
│  ○ Escalate to their manager                                 │
│  ● Escalate to the domain risk lead                          │
│  ○ Hold until reviewed (no time limit)                       │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 6: Output Style and Disclaimers

This maps to modules **OUT-06: Output Format Enforcer** and **OUT-04: Confidence Qualifier**.

```
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT STYLE AND DISCLAIMERS                                │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  What format should the tool's responses be in?              │
│                                                              │
│  ☑  Plain text (paragraphs)                                  │
│  ☑  Bullet points                                            │
│  ☐  Structured table                                         │
│  ☐  JSON (for system integrations)                           │
│                                                              │
│  Maximum response length:                                    │
│                                                              │
│  ○ Short (up to 100 words)                                   │
│  ● Medium (up to 300 words)                                  │
│  ○ Long (up to 800 words)                                    │
│  ○ No limit                                                  │
│                                                              │
│  Should every response include a disclaimer?                 │
│                                                              │
│  ● Yes                                                       │
│  ○ No                                                        │
│                                                              │
│  Disclaimer text:                                            │
│  [ This summary is auto-generated and must be reviewed     ] │
│  [ by a credit officer before use in any credit decision.  ] │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 7: Emergency Controls

This maps to module **OPS-05: Kill Switch**.

```
┌─────────────────────────────────────────────────────────────┐
│  EMERGENCY CONTROLS                                          │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  If something goes seriously wrong, who can shut this        │
│  tool down immediately?                                      │
│                                                              │
│  Primary kill switch owner:                                  │
│                                                              │
│  Name:  [ J. Smith                              ]            │
│  Role:  [ VP, Commercial Lending                ]            │
│  Email: [ j.smith@bank.com                      ]            │
│                                                              │
│  Backup kill switch owner:                                   │
│                                                              │
│  Name:  [ M. Johnson                            ]            │
│  Role:  [ SVP, Commercial Banking Risk          ]            │
│  Email: [ m.johnson@bank.com                    ]            │
│                                                              │
│  Should the tool also shut down automatically if errors      │
│  exceed a threshold?                                         │
│                                                              │
│  ● Yes — shut down if error rate exceeds [ 10 ]% in a       │
│    [ 1-hour ] window                                         │
│  ○ No — manual shutdown only                                 │
│                                                              │
│  Message to show users when the tool is shut down:           │
│                                                              │
│  [ This tool is temporarily unavailable. Please contact    ] │
│  [ the Commercial Lending team for assistance.             ] │
│                                                              │
│  [ ← Back ]  [ Next → ]                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Form Section 8: Review and Submit

After filling out all sections, the business user sees a plain-English summary of everything they configured. No technical jargon, no YAML, no module IDs.

```
┌─────────────────────────────────────────────────────────────┐
│  REVIEW YOUR GUARDRAIL CONFIGURATION                         │
│  ─────────────────────────────────────────────────────────── │
│                                                              │
│  Use Case: Commercial Lending - Credit Memo Summarizer       │
│  Risk Tier: HIGH                                             │
│                                                              │
│  ── ALREADY APPLIED (you cannot change these) ──────────── │
│                                                              │
│  ✓ PII is automatically detected and redacted                │
│  ✓ Prompt injection attacks are blocked                      │
│  ✓ Toxic content is filtered                                 │
│  ✓ All interactions are logged for audit                     │
│  ✓ Fair lending compliance is enforced                       │
│  ✓ Credit decision language is blocked                       │
│                                                              │
│  ── YOUR CONFIGURATION (Tier 3) ───────────────────────── │
│                                                              │
│  SCOPE: Tool will ONLY respond to questions about credit     │
│  memo summarization and financial statement analysis.        │
│  Off-topic requests receive: "This tool only summarizes      │
│  credit memos. Please contact your credit officer."          │
│                                                              │
│  DATA SOURCES: Credit memo corpus, financial policy docs.    │
│  Citations are required in every response.                   │
│                                                              │
│  ACCURACY: 85% of claims must be supported by source docs.   │
│  Responses below threshold are flagged for human review.     │
│                                                              │
│  HUMAN REVIEW: Required when confidence is below 70%.        │
│  Reviewer: Credit Analyst. Escalation after 4 hours to       │
│  domain risk lead.                                           │
│                                                              │
│  OUTPUT: Plain text or bullet points, up to 300 words.       │
│  Every response includes: "This summary is auto-generated    │
│  and must be reviewed by a credit officer before use in      │
│  any credit decision."                                       │
│                                                              │
│  EMERGENCY: Kill switch held by J. Smith (primary) and       │
│  M. Johnson (backup). Auto-shutdown if error rate > 10%      │
│  in 1 hour.                                                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ This configuration will be sent to your domain risk  │    │
│  │ lead for review (required for HIGH risk use cases).  │    │
│  │ You will be notified when it is approved.            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  [ ← Back to Edit ]          [ Submit for Review → ]        │
└─────────────────────────────────────────────────────────────┘
```

### What Happens After the Business User Clicks "Submit"

The business user's job is done after they submit the form. Here is what happens next, and who is responsible for each step:

```
Business User clicks "Submit for Review"
         │
         ▼
┌─────────────────────────────────────────────┐
│  TRANSLATION ENGINE (automated)              │
│                                              │
│  1. Converts form answers into technical     │
│     guardrail config (YAML/JSON)             │
│  2. Merges with Tier 1 + Tier 2 configs      │
│  3. Validates: all required modules present?  │
│  4. Validates: parameters within allowed      │
│     ranges?                                   │
│  5. If validation fails → sends form back     │
│     to business user with specific guidance   │
│     (e.g., "HIGH risk requires human review   │
│     — please complete Section 5")             │
└──────────────────┬──────────────────────────┘
                   │ Validation passes
                   ▼
┌─────────────────────────────────────────────┐
│  APPROVAL WORKFLOW (human reviewers)         │
│                                              │
│  Reviewer receives:                          │
│  • The plain-English summary (same as what   │
│    the business user saw)                    │
│  • The generated technical config            │
│  • The risk assessment scores                │
│  • Flags or warnings from the validation     │
│    engine                                    │
│                                              │
│  Reviewer can:                               │
│  • Approve as-is                             │
│  • Approve with modifications (reviewer      │
│    adjusts parameters)                       │
│  • Reject with comments (sends back to       │
│    business user with feedback)              │
└──────────────────┬──────────────────────────┘
                   │ Approved
                   ▼
┌─────────────────────────────────────────────┐
│  DEPLOYMENT (platform team, automated)       │
│                                              │
│  1. Config is version-controlled (Git)       │
│  2. Config is deployed to guardrail engine   │
│  3. Pre-deployment automated tests run       │
│  4. Monitoring & alerting activated          │
│  5. Business user notified: "Your use case   │
│     is live"                                 │
└─────────────────────────────────────────────┘
```

### How Each Form Question Maps to a Technical Module

For those who want to understand the connection between what the business user sees and the underlying technical system, here is the mapping:

| Form Section (What Business User Sees) | Technical Module (What the System Creates) | Who Built the Module |
|---|---|---|
| "What should this tool be used for?" | IN-01: Topic Scope Restrictor | Platform team |
| "What data sources can the tool use?" | SYS-02: Knowledge Boundary Setter | Platform team |
| "How accurate must responses be?" | OUT-02: Factual Grounding Checker | Platform team |
| "Does a human need to review responses?" | OPS-02: Human-in-the-Loop Gate | Platform team |
| "Output style and disclaimers" | OUT-06: Output Format Enforcer + OUT-04: Confidence Qualifier | Platform team |
| "Emergency controls" | OPS-05: Kill Switch | Platform team |

The platform team builds each module once. The business user configures it many times across many use cases — without ever touching the module's code.

### Smart Defaults and Safety Nets

The portal does not just present blank forms. It includes several mechanisms to help non-technical users make good choices and prevent dangerous configurations.

#### 1. Risk-Tier-Based Smart Defaults

When a business user opens the portal, the form is **pre-filled with recommended defaults** based on their risk tier. They can adjust these but cannot go below the minimum.

| Setting | LOW Default | MEDIUM Default | HIGH Default | CRITICAL Default |
|---|---|---|---|---|
| Accuracy threshold | 70% | 75% | 85% | 95% |
| Human review | Not required | Optional | Required (when confidence is low) | Required (always) |
| Citations | Optional | Recommended | Required | Required |
| Auto-shutdown on errors | Off | Off | On (10% threshold) | On (5% threshold) |
| Response length limit | No limit | 500 words | 300 words | 200 words |

#### 2. Mandatory Fields by Risk Tier

The portal enforces **minimum guardrail requirements** per risk tier. Some form sections cannot be skipped.

| Risk Tier | Mandatory Form Sections |
|---|---|
| LOW | Scope (Section 2), Emergency contacts (Section 7) |
| MEDIUM | Scope, Data sources (Section 3), Emergency controls |
| HIGH | ALL sections are mandatory |
| CRITICAL | ALL sections mandatory + additional fields for red team testing plan, incident response plan |

If a business user tries to submit without completing a required section, the portal blocks submission and shows a clear message:

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠  Cannot submit: your use case is rated HIGH risk.        │
│                                                              │
│  The following required sections are incomplete:             │
│                                                              │
│  • Section 5: "Does a human need to review responses?"       │
│    → HIGH risk use cases require at least one human          │
│      review trigger. Please complete this section.           │
│                                                              │
│  • Section 3: "What data sources can the tool use?"          │
│    → You must select at least one approved data source.      │
│                                                              │
│  [ Go to Section 5 ]    [ Go to Section 3 ]                 │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Guided Tooltips and Explanations

Every form field has a "What does this mean?" tooltip written in plain English with concrete examples from banking. For instance:

```
  Accuracy threshold:  ● Strict (85%)

      ┌──────────────────────────────────────────────┐
      │  ℹ  What does 85% accuracy mean?              │
      │                                               │
      │  If the tool generates a summary with 10       │
      │  factual claims, at least 8-9 of them must     │
      │  be directly supported by the source credit    │
      │  memo. If fewer than 85% are supported, the    │
      │  response is flagged for human review.         │
      │                                               │
      │  Example: If the tool says "Revenue grew 12%"  │
      │  but the memo says 10%, that claim fails the   │
      │  grounding check.                              │
      └──────────────────────────────────────────────┘
```

#### 4. Conflict Detection

The portal checks for contradictions in the business user's configuration and alerts them:

| Conflict | Portal Response |
|---|---|
| User selected "No human review" but risk tier is HIGH | "HIGH risk use cases require human review. Please select a review trigger." |
| User set accuracy to 70% but selected "regulatory reference library" as a data source | "Caution: When using regulatory data, we recommend at least 85% accuracy to avoid compliance risks. Would you like to increase it?" |
| User checked "credit decisioning" in both allowed AND blocked topics | "You selected credit decisioning as both an allowed and blocked topic. Please resolve this conflict." |

### Frequently Asked Questions from Business Teams

**Q: Do I need to know what YAML or JSON is?**
A: No. You will never see YAML, JSON, or any code. You fill out forms with checkboxes, dropdowns, and text fields. The system handles the rest.

**Q: What if I don't know which options to pick?**
A: The form pre-fills recommended defaults based on your risk tier. If you are unsure about a specific field, click the "What does this mean?" tooltip for a plain-English explanation. You can also contact the platform team's support channel for help.

**Q: Can I accidentally break something or create a security hole?**
A: No. The portal has multiple safety nets:
- Tier 1 guardrails (PII protection, toxicity filters, etc.) are always on and cannot be turned off from the portal
- Tier 2 domain guardrails are always on and cannot be weakened
- The portal enforces minimum requirements for your risk tier
- A human reviewer (risk lead or 2nd Line) reviews your configuration before it goes live
- You can only make your guardrails **stricter** than the defaults, never weaker

**Q: How long does the process take from form submission to the tool going live?**
A: This depends on the approval workflow, which varies by risk tier and your organization's capacity. The form itself can be completed in a single sitting. After submission, the review and deployment process follows the timelines set by your governance team.

**Q: Can I change my configuration after the tool is live?**
A: Yes. Return to the portal, make changes, and resubmit. Changes go through the same review and approval process. All changes are version-controlled, so there is a full audit trail of what changed, when, and who approved it.

**Q: What if my use case doesn't fit neatly into the form options?**
A: Every form section includes an "Add custom" option for topics, restrictions, and other fields. If your needs are truly unique, the platform team can work with you to create a custom guardrail module and add it to the library for future reuse.

### Summary: Who Does What

| Person | What They Do | Technical Skill Required |
|---|---|---|
| **Platform / Tech Team** | Builds the portal, builds the guardrail modules, maintains the translation engine, deploys configs, runs the infrastructure | High — software engineering |
| **Domain Risk Lead** | Defines Tier 2 domain guardrail requirements, reviews MEDIUM+ configurations, maintains domain templates | Medium — risk expertise, no coding |
| **Business Use Case Owner** | Fills out the form, selects options, writes plain-English descriptions, submits for review, monitors their use case | **None — just fill out the form** |
| **2nd Line Risk** | Reviews HIGH/CRITICAL configurations, challenges risk assessments, monitors enterprise metrics | Medium — risk expertise, no coding |

The entire point of this architecture is captured in one sentence: **the platform team builds it once, and business teams configure it forever — without writing a single line of code.**

---

## 7. Risk Assessment Framework

### What Is It?

The Risk Assessment Framework is the structured process for evaluating each GenAI use case before it goes live. Its primary output is a **risk tier** (Low, Medium, High, Critical), which determines:

- Which guardrail modules are required
- Who must approve the use case
- How intensively it must be monitored
- How often it must be recertified

### The Six Risk Dimensions

Each use case is scored across six dimensions on a **1 to 5 scale** (1 = Low Risk, 5 = Critical Risk). These dimensions capture the most important risk factors for GenAI in banking.

#### Dimension 1: Data Sensitivity

*What kind of data does the model access or could users provide?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | Public data only | Market data, published research, public filings |
| 2 | Internal non-sensitive data | General policies, procedures, org charts |
| 3 | Confidential business data | Financial reports, strategy documents, M&A data |
| 4 | Customer PII / NPPI | Names, addresses, account balances, transaction history |
| 5 | Highly regulated data | Credit bureau data, SAR/BSA data, MNPI (material non-public information), health information |

**Key questions to ask:**
- What data does the model access through its knowledge base or RAG pipeline?
- What data could users paste into prompts (even if they shouldn't)?
- Where does the training or retrieval corpus come from?
- Is the data subject to specific regulations (GLBA, FCRA, HIPAA)?

#### Dimension 2: Decision Impact

*Does the model's output influence or drive decisions?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | Informational / productivity only | Summarizing meeting notes, drafting internal emails |
| 2 | Informs decisions with significant human oversight | Research assistant where analyst independently verifies |
| 3 | Directly shapes decisions (humans typically accept the output) | Draft credit recommendations that officers usually approve |
| 4 | Semi-automated decisions with human approval step | Auto-generated risk scores reviewed by committee |
| 5 | Autonomous or near-autonomous decisions | Automated trading signals, auto-adjudicated claims |

**Key questions to ask:**
- Does the output feed into a decision-making process?
- How much does the human reviewer actually challenge the output vs. rubber-stamp it?
- What is the financial or customer impact if the output is wrong?
- Is there "automation bias" risk — will users trust the model too much?

#### Dimension 3: Customer / Consumer Impact

*Can the model's output affect customers?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | No customer exposure; purely internal | Internal code review assistant |
| 2 | Internal but output may indirectly reach customers | Drafted customer emails that are reviewed before sending |
| 3 | Customer-facing, low consequence | General FAQ chatbot with human escalation |
| 4 | Customer-facing, moderate consequence | Product recommendations, account information display |
| 5 | Customer-facing, high consequence | Credit decisions, pricing, complaint responses, legal disclosures |

**Key questions to ask:**
- Does the customer see the model's output directly?
- Can the output affect customer outcomes (approval/denial, pricing, service quality)?
- Are vulnerable populations involved (elderly, low-income, limited English proficiency)?
- What happens if the output is wrong — inconvenience or material harm?

#### Dimension 4: Regulatory & Compliance Exposure

*Which regulations apply and how actively are they enforced?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | No specific regulatory implications | Internal developer tools |
| 2 | General operational risk requirements | Internal productivity tools subject to standard ORM |
| 3 | Subject to specific regulation, low enforcement focus | Model documentation requirements (SR 11-7) |
| 4 | Subject to active regulatory scrutiny | Fair Lending (ECOA/HMDA), UDAAP, BSA/AML, fiduciary obligations |
| 5 | Directly in scope of consent orders, MRAs/MRIAs | Areas under active examination or remediation |

**Key questions to ask:**
- Which specific regulations apply to this use case?
- Is this area currently under regulatory scrutiny or examination?
- Are there pending consent orders, MRAs, or MRIAs in this domain?
- Would a failure in this use case be reportable to a regulator?

#### Dimension 5: Scale & Reach

*How many users and how broad is the impact if something goes wrong?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | Single user / pilot / proof of concept | One analyst testing a new tool |
| 2 | Small team (< 20 users), limited scope | A single team in one office |
| 3 | Department-wide (20–200 users) | Entire compliance department |
| 4 | Business-line-wide (200–2,000 users) | All commercial lending officers |
| 5 | Enterprise-wide or external-facing at scale (2,000+ users or external customers) | Customer-facing chatbot serving millions |

**Key questions to ask:**
- How many internal users will use this tool?
- How many customers could be affected?
- How many interactions per day are expected?
- What is the "blast radius" if the tool fails or produces bad output?

#### Dimension 6: Model Dependency & Overridability

*How dependent is the process on the model, and how easy is it to catch and fix errors?*

| Score | Criteria | Banking Examples |
|-------|----------|-----------------|
| 1 | Output easily verified; alternative workflows exist | Summarization tool where the source document is right there |
| 2 | Output helpful but users routinely validate independently | Research assistant where analysts cross-check |
| 3 | Output relied upon but can be overridden with moderate effort | Draft reports that would take significant time to redo manually |
| 4 | Output deeply embedded in workflow; overriding is difficult | Integrated into loan origination system with complex data flows |
| 5 | Output is the sole input to a process; no practical alternative | Fully automated pipeline with no manual fallback |

**Key questions to ask:**
- What happens if the tool goes completely down? Is there a manual fallback?
- Do users have the expertise to identify bad output?
- How long would it take to do this task without the model?
- Is there a risk that users become dependent and stop verifying?

### Risk Tier Calculation

Add up the scores from all six dimensions (range: 6 to 30):

| Total Score | Risk Tier | What It Means |
|-------------|-----------|---------------|
| **6–10** | **LOW** | Minimal risk. Self-service approval. Tier 1 guardrails sufficient. Automated monitoring. Annual recertification. |
| **11–17** | **MEDIUM** | Moderate risk. Manager + domain risk lead approval. Tier 1 + Tier 2 guardrails required. Quarterly monitoring review. Semi-annual recertification. |
| **18–23** | **HIGH** | Significant risk. 2nd Line of Defense review required. Tier 1 + Tier 2 + mandatory Tier 3 guardrails. Monthly monitoring review. Quarterly recertification. |
| **24–30** | **CRITICAL** | Highest risk. Risk committee approval. Full guardrail suite plus custom controls. Continuous real-time monitoring. Pre-deployment red team testing required. Monthly recertification. |

### Override Rule

**Any single dimension scored at 5 automatically elevates the use case to at minimum HIGH tier, regardless of the total score.**

Why? A use case could score 1 on five dimensions but 5 on "Regulatory Exposure" (e.g., it's in scope of a consent order). The total score of 10 would classify it as LOW — but it clearly isn't. The override rule prevents this.

### Worked Example

**Use Case:** Customer-facing chatbot for retail banking general inquiries

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Data Sensitivity | 4 | Customers may share account info in chat; bot accesses account data |
| Decision Impact | 3 | Bot provides product recommendations that customers act on |
| Customer Impact | 4 | Directly customer-facing, could affect product choices |
| Regulatory Exposure | 4 | UDAAP (can't be deceptive/unfair), advertising rules |
| Scale & Reach | 5 | Serves millions of retail customers |
| Model Dependency | 3 | Fallback to human agents exists but is capacity-constrained |

**Total Score: 23 → HIGH tier**

**Override triggered:** Scale & Reach = 5, so minimum HIGH tier applies (consistent with the score-based classification in this case).

**Required actions:**
- 2nd Line review before deployment
- Tier 1 + Tier 2 (Retail Banking domain) + Tier 3 guardrails
- Monthly monitoring review
- Quarterly recertification
- Adversarial testing before go-live

---

## 8. Governance Process

### End-to-End Process Flow

This is the step-by-step process for taking a GenAI use case from idea to production with proper risk governance. Note: Step 5 is where the Tier 3 Self-Service Portal (Section 6) fits in — the business user fills out the forms described there.

```
Step 1: USE CASE REGISTRATION
│  Owner submits: description, intended users, data sources,
│  expected volume, business justification
│
▼
Step 2: SELF-ASSESSMENT
│  Owner scores all 6 risk dimensions using a guided
│  questionnaire with examples and decision trees
│
▼
Step 3: AUTOMATED VALIDATION
│  System cross-checks: Are scores consistent with metadata?
│  Example: If data source = customer database, then
│  Data Sensitivity must be >= 4. Flags inconsistencies.
│
▼
Step 4: RISK TIER ASSIGNMENT
│  System calculates total score, applies override rules,
│  and assigns the risk tier (Low / Medium / High / Critical)
│
▼
Step 5: GUARDRAIL MAPPING (via Tier 3 Self-Service Portal)
│  System auto-applies Tier 1 + Tier 2 modules for the domain.
│  Business user opens the Self-Service Portal (Section 6),
│  fills out the guided forms to configure Tier 3 guardrails.
│  No coding required — just forms, dropdowns, and checkboxes.
│
▼
Step 6: REVIEW & CHALLENGE
│  LOW      → Auto-approved if validation passes
│  MEDIUM   → Domain risk lead reviews and approves
│  HIGH     → 2nd Line of Defense reviews and approves
│  CRITICAL → Risk committee reviews with 2nd Line recommendation
│
▼
Step 7: PRE-DEPLOYMENT TESTING
│  LOW/MEDIUM  → Standardized automated test suite
│  HIGH        → Standardized + targeted adversarial testing
│  CRITICAL    → Full red team exercise
│
▼
Step 8: APPROVAL & DEPLOYMENT
│  Conditional approval with specific monitoring requirements
│  documented. Guardrail configuration deployed.
│
▼
Step 9: ONGOING MONITORING & RECERTIFICATION
   Per risk tier schedule. Reassessment triggered by:
   model changes, scope changes, incidents, regulatory changes
```

### What Triggers a Reassessment?

Even after deployment, certain events should trigger a fresh risk assessment:

| Trigger | Why |
|---------|-----|
| The underlying LLM model is updated or changed | New model may behave differently |
| The use case scope expands (new features, new users, new data sources) | Risk profile has changed |
| A guardrail failure or incident occurs | Need to investigate and strengthen controls |
| A new regulation or regulatory guidance is issued | Compliance requirements may have changed |
| The use case moves from pilot to production | Scale changes the risk profile |
| Scheduled recertification date arrives | Periodic review to catch drift |

---

## 9. Risk-Tiered Oversight

### Why Tiered Oversight?

Your 2nd Line risk team cannot deeply review every single one of thousands of use cases. You must allocate your limited resources to where they matter most. Tiered oversight ensures the highest-risk use cases get the most attention, while lower-risk ones are managed efficiently through automation and self-service.

### Oversight Model

| Risk Tier | % of Use Cases (typical) | 2nd Line Involvement | Monitoring |
|-----------|--------------------------|---------------------|------------|
| **LOW** | ~40% | No individual review. Spot-check 5-10% quarterly. Trust the automated validation. | Automated dashboards. Alert on anomalies only. |
| **MEDIUM** | ~35% | No individual review by default. Sample-based review (~20% randomly per quarter). | Quarterly monitoring review by domain risk lead. |
| **HIGH** | ~20% | Full 2nd Line review before deployment. Named risk reviewer assigned. | Monthly monitoring review. Direct escalation path for incidents. |
| **CRITICAL** | ~5% | Risk committee approval. Dedicated risk oversight. Ongoing 2nd Line engagement. | Continuous real-time monitoring. Weekly metrics review. |

### Where to Focus Your Attention

**As a 2nd Line risk team, prioritize these activities:**

1. **Review all HIGH and CRITICAL use cases thoroughly** — This is approximately 25% of use cases but represents ~80% of the risk.

2. **Challenge risk assessment accuracy** — The most common gaming is underrating Decision Impact. Use case owners claim "a human reviews every output" but in practice the human rubber-stamps. Probe this.

3. **Validate guardrail effectiveness** — Don't just check that guardrails are configured. Test that they actually work by reviewing adversarial test results.

4. **Monitor Tier 1 and Tier 2 guardrail performance** — Track intervention rates across all use cases. A sudden drop in Tier 1 PII detections might mean the filter is broken, not that users stopped sharing PII.

5. **Maintain domain Tier 2 templates** — Keep them current with regulatory changes and lessons learned from incidents.

### Red Flags in Risk Assessments

Watch for these warning signs when reviewing risk assessments:

| Red Flag | What It May Indicate |
|----------|---------------------|
| Customer-facing use case scored below HIGH | Owner is underrating Customer Impact or Scale |
| "No regulatory implications" for a use case in a regulated process | Owner doesn't understand regulatory scope |
| Decision Impact scored at 2 ("significant human oversight") without evidence | Automation bias risk; human may not actually be reviewing |
| No kill switch or fallback plan documented | Operational resilience gap |
| Grounding threshold set very low (< 70%) | Owner prioritizing availability over accuracy |
| Scoring the same as a similar use case without independent analysis | Copy-paste risk assessment without genuine evaluation |

---

## 10. Continuous Improvement Loop

### Why Continuous Improvement?

GenAI risk management is not a "set it and forget it" exercise. Models change, regulations evolve, threats become more sophisticated, and new failure modes are discovered. Your guardrail framework must evolve continuously.

### The Improvement Cycle

```
        ┌─────────────┐
        │   DEPLOY     │
        │  Use Case    │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   MONITOR    │ ← Track guardrail metrics,
        │              │   user feedback, incidents
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   DETECT     │ ← Identify new failure modes,
        │              │   policy gaps, emerging risks
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   IMPROVE    │ ← Update modules, add new ones,
        │              │   tighten thresholds, update Tier 2
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  RE-DEPLOY   │ ← Push updated guardrails
        │              │   across affected use cases
        └──────┬──────┘
               │
               └──────────► (back to Monitor)
```

### Key Activities

**1. Red Teaming / Adversarial Testing**

- Require use case owners to run standardized adversarial test suites before go-live
- The platform team provides these test suites — they test for jailbreaking, PII extraction, topic boundary violations, bias elicitation, and more
- For HIGH and CRITICAL use cases, dedicated red team exercises with creative, sophisticated attacks
- Results feed back into guardrail improvements

**2. Incident Feedback Loop**

- When a guardrail failure occurs (model produces harmful output, PII is leaked, regulatory violation, etc.), conduct a root cause analysis
- Determine whether the failure indicates a gap in the Tier 1, Tier 2, or Tier 3 guardrails
- Update the relevant guardrail module or create a new one
- Push the update across all affected use cases

**3. Metrics and Reporting**

Track these metrics at the enterprise, domain, and use case level:

| Metric | What It Tells You |
|--------|-------------------|
| Guardrail trigger rate | How often guardrails are intervening. Too high = misconfigured or high abuse. Too low = potentially not working. |
| Trigger rate by module | Which types of risks are most common. Helps prioritize improvements. |
| False positive rate | How often guardrails block legitimate interactions. Too high = overly restrictive, hurting user adoption. |
| Override rate | How often human reviewers override guardrail decisions. High override rate may mean guardrails are miscalibrated. |
| User feedback scores | Overall satisfaction with output quality. Declining scores may indicate model drift. |
| Incident count and severity | Direct measure of guardrail effectiveness. |
| Time to detect and respond | How quickly issues are identified and resolved. |
| Recertification compliance | % of use cases that are current on their recertification schedule. |

---

## 11. Roles and Responsibilities

### The Three Lines of Defense Applied to GenAI

Banking uses the **Three Lines of Defense** model for risk governance. Here is how it maps to GenAI guardrail management:

| Line | Role | GenAI Guardrail Responsibility |
|------|------|-------------------------------|
| **1st Line** (Business / Use Case Owners) | Owns the risk | Registers use cases. Completes risk assessments. Configures Tier 3 guardrails. Runs pre-deployment tests. Monitors their use case. Responds to incidents. Recertifies on schedule. |
| **2nd Line** (Risk Management — your team) | Oversees and challenges | Sets the guardrail framework and standards. Maintains Tier 2 domain templates. Reviews HIGH/CRITICAL use cases. Challenges risk assessments. Monitors enterprise-wide metrics. Reports to risk committee. Updates framework based on incidents and regulatory changes. |
| **3rd Line** (Internal Audit) | Provides independent assurance | Audits the framework's design and operating effectiveness. Tests whether guardrails work as intended. Validates that risk assessments are accurate. Reports to audit committee and board. |

### Platform / Technology Team

The platform team is not a "line of defense" but is a critical enabler:

- Builds and maintains Tier 1 universal guardrails
- Builds the Guardrail Template Library (modules)
- Builds the self-service configuration portal
- Builds the merge/deployment pipeline
- Builds monitoring dashboards and alerting
- Provides adversarial test suites
- Manages model infrastructure and security

---

## 12. Summary

### The Five Key Principles

1. **Centralize the universal baseline (Tier 1)** — The platform team builds and enforces protections that every use case needs. No exceptions, no opt-outs.

2. **Standardize domain-level policies (Tier 2)** — Each business domain has pre-defined guardrails reflecting its specific regulatory and risk environment. Automatically applied, can be tightened but not loosened.

3. **Democratize use-case-specific configuration (Tier 3)** — Give use case owners a self-service toolkit of pre-built modules. They configure, not code. They know their use case best.

4. **Risk-tier your oversight** — Not all use cases are equal. Focus your scarce 2nd Line resources on the 20-25% of use cases that carry the most risk. Automate oversight of the rest.

5. **Continuously improve** — Monitor, detect, improve, redeploy. The framework is alive. Feed incidents, test results, regulatory changes, and metrics back into guardrail improvements.

### What Makes This Approach Work for Banking

| Banking Requirement | How This Framework Addresses It |
|--------------------|---------------------------------|
| SR 11-7 Model Risk Management | Risk assessment framework integrates with model risk inventory; tiered oversight aligns with model risk tiering |
| Regulatory expectations (OCC, Fed, FDIC) | Documented governance, risk assessment, testing, monitoring, and independent review |
| Fair Lending / ECOA | Tier 2 guardrails for lending include bias detection, prohibited language, decision boundary enforcement |
| BSA/AML | Tier 2 guardrails for AML/KYC include SAR confidentiality, prohibited disclosures |
| UDAAP | Content filters, tone enforcement, accuracy requirements prevent unfair/deceptive practices |
| Data privacy (GLBA, CCPA) | Tier 1 PII detection, data classification, audit logging |
| Operational resilience | Kill switches, fallback plans, monitoring, incident response |
| Audit trail | Comprehensive logging at all tiers, version-controlled configurations |

---

*This guide provides a framework. Each bank should adapt the specific modules, scoring criteria, thresholds, and governance processes to fit their organization's size, complexity, risk appetite, and regulatory environment.*
