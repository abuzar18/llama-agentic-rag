# Northwind Labs Security and AI Usage Policy (sample data)

This policy is fictional sample data for testing the RAG pipeline.

## Data classification

Company data has three levels: Public, Internal and Confidential. Customer personal data (names, emails, phone numbers, health or payment information) is always Confidential. Confidential data must be encrypted at rest and in transit and may only be stored in approved systems.

## Using AI tools

Employees may use approved AI assistants for drafting, coding and research. Confidential data must never be pasted into an AI tool unless that tool is listed as approved for Confidential data by the security team. Personally identifiable information (PII) must be redacted before any document is sent to an external model API. All AI-generated code must be reviewed by a human before it is merged.

## Model and prompt changes

Production prompts and model versions are version-controlled. Any change to a production prompt or model requires an evaluation run on the regression test set and approval from a second engineer. Evaluation results must be attached to the pull request.

## Incident response

Security incidents must be reported to security@northwind.example within 1 hour of discovery. The on-call engineer acknowledges incidents within 15 minutes during business hours. Severity 1 incidents (customer data exposure or full outage) trigger a status-page update within 30 minutes and a written post-incident review within 5 working days.

## Access control

Access to production systems requires single sign-on and hardware-key multi-factor authentication. Access is reviewed every quarter and removed within 24 hours when an employee leaves the company.
