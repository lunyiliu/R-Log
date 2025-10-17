## Log Parsing

### 🧠 Conceptional Thinking Templates

**Step 1:** Replace double spaces with a single space.  
**Step 2:** Replace digit tokens with variables.  
**Step 3:** Replace `True` / `False` with a variable.  
**Step 4:** Replace a path-like token with a variable.  
**Step 5:** Replace a token containing both fixed and variable parts with a variable.  
**Step 6:** Replace consecutive variables as a single variable.  
**Step 7:** Replace dot-separated variables as a single variable.  

### ⚙️ Procedural Thinking Templates

**Step 1:** Replace sequences of numbers with `<*>`.  
**Step 2:** Replace IP addresses with `<*>`.  
**Step 3:** Replace host names with `<*>`.  
**Step 4:** Replace any occurrence of `size ` with `size <*>`.  
**Step 5:** Replace any occurrence of `blk_` with `<*>`.  
**Step 6:** Replace any sequence indicating a user with `user=<*>`.  
**Step 7:** Replace any `uid=` and `euid=` with `uid=<*>` and `euid=<*>` respectively.  
**Step 8:** Keep the structure and text for logs that have no variable parts.  
**Step 9:** Replace time indicators like ` s` with `<*> s`.  
**Step 10:** For statements resembling an event or action with a number, replace the number with `<*>`.  
**Step 11:** For logs indicating a connection or disconnection, replace specific addresses and ports with `<*>`.  

---

## Log Anomaly Detection

### 🧠 Conceptional Thinking Templates

**Step 1:** Mark it normal when values (such as memory address, floating number and register value) in a log are invalid.
**Step 2:** Mark it normal when there is a lack of information.
**Step 3:** Never consider ⟨*⟩and missing values as abnormal patterns.
**Step 4:**  Mark it abnormal when and only when the alert is explicitly expressed in textual content (such as keywords like error or interrupt).

### ⚙️ Procedural Thinking Templates for BGL

**Step 1:** Check diagnostics such as register dumps, counters, configuration reports, health metrics, and service banners, which are routine and normal.
**Step 2:** Review fault-tolerance signals like correctable errors, link retries, transient exceptions, and management tool messages, which are normal if recovery occurs.
**Step 3:**  Watch for kernel crashes, assertion failures, or abnormal exits of core processes, which indicate unrecoverable software or runtime errors that cause abnormal behavior.
**Step 4:** Examine power modules, machine check interrupts, network or interconnect losses, and storage mount failures, which may signal hardware defects or service disruption that cause abnormal behavior.

### ⚙️ Procedural Thinking Templates for Spirit

**Step 1:** Check system initialization and hardware configuration outputs (CPU registers, APIC/LAPIC, RAID/partition checks, BIOS/PCI corrections, memory maps) which are routine during boot or reconfiguration. 
**Step 2:** Review service and driver messages (Postfix retries, daemons, device drivers, HP agents, RAID detection, logging daemons) which are expected and normal unless critical errors appear. 
**Step 3:**  Observe fault-tolerance related outputs (checkpoint warnings, NFS mount permission errors, deferred mail, login/authentication retries, memory allocation warnings) which may indicate recoverable issues if the system continues to run.  
**Step 4:** Watch for unrecoverable conditions (NFS failures, disk/controller errors like drive not ready, failed epilog jobs, assertion failures, repeated root authentication failures) which indicate abnormal behavior requiring attention.  
**Step 5:** Examine hardware or network faults (machine check errors, LANai not running, node bailout failures, repeated protocol errors) which suggest hardware defects, instability, or service disruption causing abnormal behavior. 

---

## Log Interpretation

### 🧠 Conceptional Thinking Templates

**Step 1:** Contextualize the log entry by considering the system environment, application type, and operational context to understand the broader scenario.
**Step 2:** Identify the error type and its general implications using domain knowledge and common patterns in software behavior.
**Step 3:** Relate the error to specific system components, configurations, or code paths that could generate such an entry, leveraging mental models of how errors propagate.
**Step 4:** Synthesize an interpretation that explains the error's significance, ensuring it aligns with the observed log data and avoids assumptions without evidence.

### ⚙️ Procedural Thinking Templates

**Step 1:** Extract the exact error message, code, or keyword from the log (e.g., "500 Internal Server Error").
**Step 2:** Classify the error based on category (e.g., server-side, client-side, network) using predefined taxonomies or experience.
**Step 3:** Recall standard documentation, common causes, or known issues associated with the error (e.g., PHP configuration directives like display_errors).
**Step 4:** Examine additional log details such as timestamps, severity levels, stack traces, or environment variables for contextual clues.
**Step 5:** Formulate a concise interpretation by combining the error message with contextual information, ensuring it addresses the user's query directly.

---

## Root Cause Analysis

### 🧠 Conceptional Thinking Templates

**Step 1:** Isolate the key event or error from the log that represents the symptom of the problem, focusing on its uniqueness and impact.
**Step 2:** Generate multiple hypotheses for potential root causes by considering system dependencies, recent changes, and failure modes from past incidents.
**Step 3:** Validate each hypothesis against log evidence, such as error sequences, frequency, or correlations, using deductive reasoning.
**Step 4:** Converge on the most plausible root cause by applying principles like Occam's razor or prioritizing causes with the highest explanatory power.

### ⚙️ Procedural Thinking Templates

**Step 1:** Identify the primary error or failure point in the log, noting any repeated patterns or anomalies.
**Step 2:** Trace backward in the log to find preceding events that might have triggered the error (e.g., configuration updates, resource usage spikes, dependency failures).
**Step 3:** Check for common root causes such as software bugs, misconfigurations, hardware issues, or version incompatibilities, based on industry best practices.
**Step 4:** Analyze specific error details for clues (e.g., PHP version changes, directive settings like error_reporting), and cross-reference with known documentation.
**Step 5:** FConfirm the root cause by matching log evidence with known issues, ensuring it fully explains the observed behavior without contradictions.

### 📖 Management Guidance Templates

**Step 1:** Recognize the problem by distinguishing the triggering event from the real underlying issue.
**Step 2:** Evaluate the significance of the problem by evaluating severity, likelihood, and recurrence and whether it reflects systemic or cultural deficiencies.
**Step 3:** Determine the immediate causes (conditions or actions) that directly led to the problem.
**Step 4:**  Investigate the reasons why the causes in the preceding identification step existed, working your way back to the root cause.

---

## Solution Recommendation

### 🧠 Conceptional Thinking Templates

**Step 1:** Derive potential solutions from the identified root cause, considering both immediate fixes and long-term improvements.
**Step 2:** Evaluate each solution's feasibility, risk, and impact on the system, prioritizing those that are minimal, tested, and sustainable.
**Step 3:** Select the most appropriate solution based on cost-benefit analysis, alignment with system constraints, and organizational policies.
**Step 4:** Provide a recommendation with clear references to official sources, community wisdom, or best practices to ensure credibility.

### ⚙️ Procedural Thinking Templates

**Step 1:** Review the root cause analysis to determine the exact action required (e.g., change configuration, apply patch, update software).
**Step 2:** Search for official documentation, release notes, or community forums for verified solutions (e.g., PHP manual for display_errors setting).
**Step 3:** Specify the solution steps precisely (e.g., "Set display_errors to On in php.ini" or "Update PHP to a newer version").
**Step 4:** Include references to specific resources (e.g., PHP changelog for version 5.2.4) to provide evidence and context.

### 📖 Management Guidance Templates

**Step 1:** Identify the corrective action for each cause.
**Step 2:** Assess whether the corrective actions can prevent recurrence.
**Step 3:** Determine whether the corrective actions are feasible.
**Step 4:** Verify whether the corrective actions allow meeting primary objectives or mission.
**Step 5:** Examine whether the corrective actions avoid introducing new risks, with any assumed risks clearly stated and without degrading the safety of other systems. 
**Step 6:** Review whether the immediate actions taken were appropriate and effective.
