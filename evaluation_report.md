# PetroChat - RAG Evaluation Report (2026-05-25 08:57:13)

This report details the automated accuracy evaluation of the PetroChat RAG pipeline using Ground Truth factual and unanswerable questions.

## 📈 Performance Summary

- **Overall Pass Rate**: `100.0%` (6/6 cases)
- **Factual Query Accuracy**: `100.0%` (4/4 cases)
- **Guardrail Rejection Rate**: `100.0%` (2/2 cases)
- **Average LLM Generation Latency**: `0.66 seconds` 

## 🔍 Detailed Test Case Results

### Case 1 (factual): What are the OSHA safety requirements for tank gauging operations?

- **Result Status**: ✅ PASS
- **Latency**: `0.92s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `['[OSHA 3843, Page 6]']`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: The Generated Answer contains all key factual information present in the Expected Answer, with minor stylistic and phrasing differences that do not affect the semantic equivalence or factual accuracy.

#### Generated Answer:
> The OSHA safety requirements for tank gauging operations include:
* Implementing alternative tank gauging and sampling procedures to minimize worker exposure to hydrocarbon gases and vapors [OSHA 3843, Page 6]
* Retrofitting existing tanks with dedicated sampling ports to eliminate the need to open thief hatches [OSHA 3843, Page 6]
* Reducing tank pressure prior to gauging [OSHA 3843, Page 6]
* Using appropriate respiratory protection, including supplied air respirators (SAR) and self-contained breathing apparatus (SCBA) in areas where IDLH VOC exposures may occur [OSHA 3843, Page 6]
* Providing hazard communication training in a language that employees understand [OSHA 3843, Page 6]
* Ensuring workers are trained on and correctly use calibrated multi-gas and oxygen monitors [OSHA 3843, Page 6]
* Prohibiting employees from working alone during tank gauging operations [OSHA 3843, Page 6]
* Consulting with a trained occupational safety and health professional to determine the appropriate respirator to be used [OSHA 3843, Page 6]

--- 

### Case 2 (factual): Explain the mud weight calculation and maintenance requirements for preventing a well kick.

- **Result Status**: ✅ PASS
- **Latency**: `1.10s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `['[ABB Oil & Gas Production Handbook, Page 36]', '[API RP 54, Page 52]']`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: The Generated Answer contains the key factual information present in the Expected Answer, including the importance of balancing downhole pressure and maintaining adequate volumes of kill weight fluid.

#### Generated Answer:
> To prevent a well kick, the mud weight should balance the downhole pressure to avoid leakage of gas and oil [ABB Oil & Gas Production Handbook, Page 36]. The mud weight calculation is crucial in maintaining well control. 

The following are key considerations for mud weight calculation and maintenance:
* Mud weight should be sufficient to balance the formation pressure [ABB Oil & Gas Production Handbook, Page 36].
* The mud mix is designed to match the desired flow thickness, lubrication properties, and specific gravity [ABB Oil & Gas Production Handbook, Page 36].
* A special high-density mud called “kill fluid” is used to shut down a well for workover [ABB Oil & Gas Production Handbook, Page 36].
* Adequate volumes of kill weight fluid should be on location prior to flowing the well [API RP 54, Page 52].

Maintenance requirements for preventing a well kick include:
* Ensuring the mud weight is sufficient to balance the downhole pressure [ABB Oil & Gas Production Handbook, Page 36].
* Monitoring the mud properties and adjusting as necessary [ABB Oil & Gas Production Handbook, Page 36].
* Having adequate volumes of kill weight fluid on location [API RP 54, Page 52].
* Ensuring the well test equipment and emergency shutdown system are fully operational [API RP 54, Page 52].

--- 

### Case 3 (factual): Describe the Process Safety Management (PSM) standard requirements for a refinery.

- **Result Status**: ✅ PASS
- **Latency**: `1.14s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `['[OSHA 3918, Page 5]', '[OSHA 3918, Page 19]']`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: The Generated Answer contains the key factual information present in the Expected Answer, including Process Safety Information (PSI), Process Hazards Analysis (PHA), Operating Procedures, Mechanical Integrity (MI), and Management of Change (MOC), despite minor stylistic and phrasing differences.

#### Generated Answer:
> The Process Safety Management (PSM) standard requirements for a refinery include [OSHA 3918, Page 5]:
* Process Safety Information (PSI): Employers are required to compile written process safety information to identify and understand the hazards posed by the process [OSHA 3918, Page 5].
* Process Hazards Analysis (PHA): A requirement to conduct a thorough analysis of the process hazards [OSHA 3918, Page 5].
* Operating Procedures: Employers must establish written operating procedures to ensure safe operation of the process [OSHA 3918, Page 5].
* Mechanical Integrity (MI): The Mechanical Integrity element requires employers to create written procedures to maintain the ongoing integrity of process equipment, including [OSHA 3918, Page 19]:
  + Training for process maintenance activities
  + Inspecting and testing process equipment
  + Correcting equipment deficiencies
  + Performing quality assurance
  + Addressing pressure vessels, storage tanks, piping systems, and other equipment
* Management of Change (MOC): A requirement to manage changes to the process, including equipment, procedures, and personnel [OSHA 3918, Page 5].

Refineries are also encouraged to review their safe work procedures, including [OSHA 3918, Page 19]:
* Restricting motorized equipment from ignition source controlled areas
* Ensuring that a hot work permit has been issued
* Compliance with existing prescriptive standards, such as 29 CFR 1910.146, 29 CFR 1910.147, 29 CFR 1910.252, and 29 CFR 1910.307. 

For more information, refineries can refer to OSHA's Process Safety Management Guide (OSHA 3132) or the full text of the standard at www.osha.gov [OSHA 3918, Page 5].

--- 

### Case 4 (factual): What is a blowout preventer (BOP)?

- **Result Status**: ✅ PASS
- **Latency**: `0.34s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `['[API RP 54, Page 9]']`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: The Generated Answer contains the same key factual information as the Expected Answer, with only a minor addition of a reference.

#### Generated Answer:
> A blowout preventer (BOP) is a device attached to the wellhead or tree that allows the well to be closed in with or without a string of pipe or wireline in the borehole [API RP 54, Page 9].

--- 

### Case 5 (unanswerable): What are the safety protocols for a blowout preventer (BOP) failure?

- **Result Status**: ✅ PASS
- **Latency**: `0.20s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `[]`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: Correctly rejected out-of-context query.

#### Generated Answer:
> I cannot answer this question based on the provided documents.

--- 

### Case 6 (unanswerable): What are the detailed engineering steps for designing a carbon capture system in a coal-fired power plant?

- **Result Status**: ✅ PASS
- **Latency**: `0.25s` | **Keyword Overlap**: `100.0%` 
- **Citation Check**: `Pass` (Expected: `[]`)
- **LLM Judge Evaluation**: `Pass`
  - *Reasoning*: Correctly rejected out-of-context query.

#### Generated Answer:
> I cannot answer this question based on the provided documents.

--- 
