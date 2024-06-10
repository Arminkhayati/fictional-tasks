
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'LANGCHAIN_API_KEY'

os.environ['TAVILY_API_KEY'] = 'TAVILY_API_KEY'

text = """
This 60-year-old male was hospitalized due to moderate ARDS from COVID-19 with symptoms of fever, dry cough, and dyspnea. We encountered several difficulties during physical therapy on the acute ward. First, any change of position or deep breathing triggered coughing attacks that induced oxygen desaturation and dyspnea. To avoid rapid deterioration and respiratory failure, we instructed and performed position changes very slowly and step-by-step. In this way, a position change to the 135° prone position () took around 30 minutes. This approach was well tolerated and increased oxygen saturation, for example, on day 5 with 6 L/min of oxygen from 93% to 97%. Second, we had to adapt the breathing exercises to avoid prolonged coughing and oxygen desaturation. Accordingly, we instructed the patient to stop every deep breath before the need to cough and to hold inspiration for better air distribution. In this manner, the patient performed the breathing exercises well and managed to increase his oxygen saturation. Third, the patient had difficulty maintaining sufficient oxygen saturation during physical activity. However, with close monitoring and frequent breaks, he managed to perform strength and walking exercises at a low level without any significant deoxygenation. Exercise progression was low on days 1 to 5, but then increased daily until hospital discharge to a rehabilitation clinic on day 10.
"""
local_llm = 'llama3'


# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)


"""
You are a world-class algorithm for extracting information in structured formats. 
Extract the attribute values from the patient descriptions in a JSON format. 
Valid attributes are Disease, Race, Age, Symptoms list, Disease Severity, Sex, Survival Status if died or survived or unknown. 
If an attribute is not present in the product title, the attribute value is supposed to be 'n/a'.

The patient description:
{description}
"""


"""
You are a world-class algorithm for extracting information in structured formats. 
Extract the attribute values from the patient description in a JSON format. 
Explore what features or variables related to diseases and population and output a list of these variables and features without their values.

The patient description:
{description}
"""

prompt = PromptTemplate(
    template="""
        You are a world-class algorithm for extracting information in structured formats. 
        Extract demographic features names without values from the patient description in a JSON format. 
        
        The patient description:
        {description}
        """,
    input_variables=["description"],
)

model = prompt | llm | JsonOutputParser()
# description = """
# This 60-year-old male was hospitalized due to moderate ARDS from COVID-19 with symptoms of fever, dry cough, and dyspnea. We encountered several difficulties during physical therapy on the acute ward. First, any change of position or deep breathing triggered coughing attacks that induced oxygen desaturation and dyspnea. To avoid rapid deterioration and respiratory failure, we instructed and performed position changes very slowly and step-by-step. In this way, a position change to the 135° prone position () took around 30 minutes. This approach was well tolerated and increased oxygen saturation, for example, on day 5 with 6 L/min of oxygen from 93% to 97%. Second, we had to adapt the breathing exercises to avoid prolonged coughing and oxygen desaturation. Accordingly, we instructed the patient to stop every deep breath before the need to cough and to hold inspiration for better air distribution. In this manner, the patient performed the breathing exercises well and managed to increase his oxygen saturation. Third, the patient had difficulty maintaining sufficient oxygen saturation during physical activity. However, with close monitoring and frequent breaks, he managed to perform strength and walking exercises at a low level without any significant deoxygenation. Exercise progression was low on days 1 to 5, but then increased daily until hospital discharge to a rehabilitation clinic on day 10.
# """

# description = """A 24-year-old healthy woman presented with difficulty breathing and dissatisfaction with her facial appearance. She had a history of childhood trauma resulting in nasal septum deviation and external nasal deformity. Four months after a successful and uneventful septorhinoplasty, she presented to the emergency department with blunt nasal trauma resulting in a septal hematoma, which was drained successfully; the patient was discharged with no adverse sequelae.
# Four months later, the patient sustained nasal trauma again, this time accompanied by clear nasal discharge, raising suspicion of cerebrospinal fluid (CSF) leak. The patient was discharged after managing the nasal injury, as the CT brain showed an intact cribriform plate with no evidence of a CSF leak. Ten days later, she presented at the emergency department with dizziness and an unstable gait. She also had complaints of paresthesia for the past two months, beginning in her right hand and progressing to the right shoulder, arm and leg, associated with some difficulty in the execution of movements in the first and second finger of the right hand. Her right leg was quite stiff with difficulty in walking. On close inquiry, she gave history of pain in the right eye and double vision many months back, which had resolved spontaneously. Examination showed a positive Rombergâ€™s and Lhermitteâ€™s sign, with right-sided sensory impairment.
# Magnetic resonance imaging (MRI) of the brain, cervical and thoracic spine demonstrated demyelinating lesions in the brain and cervical segment of the spinal cord (Figure ). Some of the lesions demonstrated enhancement on post gadolinium administration sequences, suggestive of active demyelinating diseases like MS. A lumbar puncture was performed which demonstrated the presence of oligoclonal bands in the CSF. The diagnosis of MS was confirmed by a neurologist and treatment was initiated.
# The initial neurological symptoms have largely vanished with only persistent light paresthesia in the right hand. Two years later she has had no new symptoms and continues with the same medication with good tolerance."""

description ="""A 64-year-old Caucasian male smoker with a horseshoe kidney with a history of open pyelolithotomy 18 years ago, presented to King Abdulaziz Medical City in mid-2020 with a report from another hospital stating that he developed gross hematuria six months prior, which was treated as a urinary tract infection. A CT of the abdomen and pelvis was performed in that hospital, showing a horseshoe kidney with severe left hydronephrosis and enlarged retroperitoneal lymph nodes, with the largest one located in the posterior part of the left renal artery measuring 4.7 Ã— 3.5 Ã— 2.6 cm. Additionally, there were multiple stones (Figures , , ). Urine culture was performed and revealed that various organisms were isolated (10-100,000 CFU/ml). Urinalysis showed a small amount of blood with a moderate presence of leukocytes and a trace protein.
At the end of 2020, the patient underwent magnetic resonance imaging (MRI). The MRI showed a horseshoe kidney with chronic hydronephrosis of the left kidney and a large mass within it centrally with further satellite lesions, which all likely represent UC and associated lymphadenopathy along the para-aortic chain (Figure ). Additionally, a finding of chronic pancreatitis was noted with dilated duct and stone, for which the patient was referred to the gastroenterology department. Furthermore, a bone scan and chest CT were performed, and no significant abnormality or metastasis was found.
After a couple of days, the patient presented to the emergency department with non-radiating progressive lower abdominal and left colicky flank pain for three days with hematuria and constipation with fullness. The patient denied any history of fever or vomiting. There were no other genitourinary symptoms, scrotal pain, or change in the level of consciousness. Vital signs were measured and were as follows: blood pressure, 151/71 mmHg; heart rate, 109; respiratory rate, 20; and temperature, 37.1â„ƒ. The weight of the
"""
print(model.invoke({"description": description}))