"""Every in-context example of the Agent-As-Tool port, in the one file the paper's appendix prints.

The paper's own, byte for byte. PAPER_SUPERVISOR holds the supervisor's examples of the five paper-era agents, the
FewShots files of IBM/AssetOpsBench archive/legacy-tmp 27552e9 (src/tmp/meta_agent/agents/*/), which the paper-era AgentHub
merges in its registration order into the 28 examples of IBM's own sample trajectories (tests/data).

Adapted from the paper's. The IoT agent's eight examples are BMSFEWSHOTS of src/tmp/IoTAgent/BMSFewShots.py on the same
branch, and the TSFM agent's six are its demonstration in IBM's sample claude-4-sonnet/0420 of branch trajectory_refactor
097b1d8. Their tool names and argument names follow the MCP servers the port reaches, and an example whose tool those
servers lack is replaced one for one by an example written for this port, so the count stays the paper's. The closing answer of the TSFM
agent's finetuning example is written here in place of the sample's.

Written for this port, where the paper has none. The supervisor's examples of TaskAgent, three in the slot of TSFMAgent, and of
CausalDiagnosisAgent, five in the slot of RuleLogicAgent, so the rig's supervisor sees 28 examples with the
CausalDiagnosis MCP and 23 without it. The FMSR agent's one example, whose paper text is not public, and the WO agent's
five, since the paper's WO agent ran in code mode. One example per tool of the task tools and of the CausalDiagnosis MCP.

Observations. In the examples written for the rig and in the IoT agent's adapted history example, every answer of an MCP
tool is a JSON object in keys that the tool's answer has, with made-up values of a made-up world: the chiller CHL-A100
with the parts compressor, condenser, evaporator and fan at SiteX, and Chiller 4 at MAIN, the paper's own names. Every
Finish line states the values that answer its question, copied from its observation, and the supervisor examples written for this port
show the reply that such a Finish line gives. The examples written for the reproduction keep their values and show the
keys of the main-0.x servers (commit c6b96755), and the examples AssetOpsBench wrote keep their text apart from the renames above.

Neutrality. Every value in an example written for this port belongs to the made-up world or is the example's own question. The diagnosis
example is written at run time from the tool contract and the registered example start of the run, a development start
and never a test start (diagnosis_example). No fixed text names a level, a channel, a reading or an evidence identifier
of a registered asset, and none says when to read, diagnose, declare, order or finish. Each question asks for one action
and the example takes it, as the paper's examples do, and one supervisor example written for this port passes the answer of one
sub-agent on to another. An example written for this port cites evidence in the forms that the surface gives.
"""
import json

# ----- the paper's, byte for byte -----

PAPER_SUPERVISOR = {
    'IoTAgent': """Question: send me the values for CU02004 at SiteX for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Thought 1: I need to download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP".
Action 1: IoTAgent
Action Input 1: request=download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "total_observations": 25, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c328516a-643f-40e6-8701-e875b1985c38.json", "message": "found 25 observations. file_path contains a JSON array of Observation data"}

Question: what sites are there
Action 1: IoTAgent
Action Input 1: request=list all the sites
Observation 1: {"sites": ["SiteX", "SiteY", "SiteZ"], "message": "List of available sites retrieved."}

Question: what assets are at site SiteX
Action 1: IoTAgent
Action Input 1: request=list assets for site SiteX
Observation 1: {"site_name": "SiteX", "assets": ["CU02004", "CU02005", "CU02006"], "message": "List of assets at SiteX retrieved."}

Question: download sensor data for CU02004 at SiteX
Action 1: IoTAgent
Action Input 1: request=download sensor data for CU02004 at SiteX
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "sensor_data": "data_path=/path/to/sensor_data.json", "message": "Sensor data for CU02004 downloaded."}

Question: download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Action 1: IoTAgent
Action Input 1: request=download asset history for CU02004 at SiteX from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00 for "CHILLED WATER LEAVING TEMP" and "CHILLED WATER RETURN TEMP"
Observation 1: {"site_name": "SiteX", "assetnum": "CU02004", "total_observations": 25, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c328516a-643f-40e6-8701-e875b1985c38.json", "message": "found 25 observations. file_path contains a JSON array of Observation data"}

Question: merge these JSON files file1.json and file2.json into a single JSON file
Action 1: IoTAgent
Action Input 1: request=merge JSON files file1.json and file2.json
Observation 1: {"file1": "/path/to/file1.json", "file2": "/path/to/file2.json", "merged_file": "/path/to/merged_file.json", "message": "Files merged successfully."}

Question: How do I get a list of properties from a JSON file
Action 1: IoTAgent
Action Input 1: request=list properties from JSON file
Observation 1: {"json_file": "/path/to/file.json", "properties": ["property1", "property2", "property3"], "message": "List of properties extracted from JSON file."}

Question: I need to read the JSON file 0001.json
Action 1: IoTAgent
Action Input 1: request=read JSON file 0001.json
Observation 1: {"file_path": "/path/to/0001.json", "content": {"key1": "value1", "key2": "value2"}, "message": "JSON file 0001.json read successfully."}

Question: how do I calculate the start date for last week or past week?
Action 1: IoTAgent
Action Input 1: request=calculate start date for last week
Observation 1: {"last_week_start_date": "2025-04-10T00:00:00", "message": "Start date for last week calculated successfully."}""",
    'TSFMAgent': """Question: Analyze the time-series data in file xxxxx.json and look for anomalies in Chiller 6 Return Temperature.
Thought 1: I need to perform time-series feature monitoring on Chiller 6 Return Temperature.
Action 1: TSFMAgent
Action Input 1: request=Analyze the time-series data in file xxxxx.json and detect anomalies in Chiller 6 Return Temperature.
Observation 1: Completed anomaly detection analysis for Chiller 6 Return Temperature.

Question: Analyze file chiller_power_data.json and identify any sudden spikes in Chiller 7 power consumption.
Thought 1: I need to analyze the time-series data for sudden spikes in power consumption.
Action 1: TSFMAgent
Action Input 1: request=Analyze file chiller_power_data.json and detect sudden spikes in Chiller 7 power consumption.
Observation 1: Completed spike detection analysis for Chiller 7 power consumption.

Question: Forecast the next 24 hours of Evaporator Leaving Water Temperature for CHL-X0 based on file evap_temp_data.json.
Thought 1: I need to build a forecast model using the time-series data.
Action 1: TSFMAgent
Action Input 1: request=Forecast next 24 hours of Evaporator Leaving Water Temperature using evap_temp_data.json for CHL-X0.
Observation 1: Completed 24-hour forecast for Evaporator Leaving Water Temperature for CHL-X0.""",
    'FMSRAgent': """Question: List all failure modes of asset CHL-A100.
Thought 1: I need to find the failure modes related to CHL-A100 by querying FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Please list all failure modes for CHL-A100.
Observation 1: The failure modes of CHL-A100 are: Compressor Overheating, Evaporator Fouling, Low Refrigerant Charge, Sensor Faults.

Question: List all installed sensors of asset CHL-A100.
Thought 1: I need to retrieve the list of installed sensors for CHL-A100 from FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Please list all installed sensors for CHL-A100.
Observation 1: Installed sensors for CHL-A100 are: Discharge Temperature Sensor, Suction Pressure Sensor, Power Meter, Vibration Sensor.

Question: List all failure modes of CHL-A100 that can be detected by temperature sensors.
Thought 1: I need to filter the failure modes of CHL-A100 based on detectability by temperature sensors using FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Which failure modes of CHL-A100 can be detected using temperature sensors?
Observation 1: Failure modes detectable by temperature sensors include: Compressor Overheating, Evaporator Fouling.

Question: If compressor overheating occurs for CHL-A100, which sensor should be prioritized for monitoring this specific failure?
Thought 1: I need to determine which sensor is most effective for detecting compressor overheating for CHL-A100 by querying FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Which sensor should be prioritized to monitor compressor overheating for CHL-A100?
Observation 1: The Discharge Temperature Sensor should be prioritized for detecting compressor overheating.

Question: When compressor motor of a chiller fails, what is the temporal behavior of the power input?
Thought 1: I need to understand the expected power input behavior during compressor motor failure by asking FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=Describe the power input behavior when the compressor motor of a chiller fails.
Observation 1: Power input typically shows a sudden drop followed by unstable fluctuations before reaching zero.

Question: When power input of CHL-A100 drops, what is the potential failure that causes it?
Thought 1: I need to find out which failure modes are associated with a drop in power input for CHL-A100 by consulting FMSRAgent.
Action 1: FMSRAgent
Action Input 1: request=What potential failures cause a drop in power input for CHL-A100?
Observation 1: A drop in power input for CHL-A100 may indicate Compressor Motor Failure, Control System Fault, or Sudden Shutdown.""",
    'WOAgent': """Question: Create a work order for inspecting CHL-A100 due to suspected compressor overheating.
Thought 1: I need to generate a work order for CHL-A100 inspection focused on compressor overheating.
Action 1: WorkOrderAgent
Action Input 1: request=Create a work order for asset CHL-A100 — task: inspect compressor for signs of overheating.
Observation 1: Work order created successfully for compressor inspection on CHL-A100.

Question: Generate a maintenance work order to replace the vibration sensor on WT-22.
Thought 1: I need to initiate a work order for replacing the vibration sensor on WT-22.
Action 1: WorkOrderAgent
Action Input 1: request=Create a work order for asset WT-22 — task: replace faulty vibration sensor.
Observation 1: Work order created for replacing vibration sensor on WT-22.

Question: Schedule a corrective work order for CHL-X0 due to detected low refrigerant charge.
Thought 1: Low refrigerant charge detected — a corrective work order is needed.
Action 1: WorkOrderAgent
Action Input 1: request=Schedule a corrective work order for asset CHL-X0 — task: address low refrigerant charge.
Observation 1: Corrective work order scheduled for CHL-X0 refrigerant charge issue.

Question: Raise an emergency work order for CHL-A100 after sudden shutdown.
Thought 1: CHL-A100 has experienced a sudden shutdown — immediate work order needed.
Action 1: WorkOrderAgent
Action Input 1: request=Create an emergency work order for CHL-A100 — task: investigate sudden shutdown.
Observation 1: Emergency work order created for sudden shutdown investigation on CHL-A100.

Question: Close the work order WO-12345 after confirming successful repair of suction pressure sensor.
Thought 1: The repair for the suction pressure sensor is completed — the work order can be closed.
Action 1: WorkOrderAgent
Action Input 1: request=Close work order WO-12345 — repair of suction pressure sensor completed successfully.
Observation 1: Work order WO-12345 closed successfully.""",
    'RuleLogicAgent': """Question: I'd be interested in setting up an alert if Chiller 6 is operational, the difference between supply temperature and liquid refrigerant evaporator temperature exceeds 4 degrees Fahrenheit, and this condition persists for at least three hours.
Thought 1: I need to generate a rule based on sensor data conditions.
Action 1: RuleLogicAgent
Action Input 1: Set up an alert for Chiller 6 when operational, the difference between supply temperature and liquid refrigerant evaporator temperature exceeds 4°F for at least three hours.
Observation 1: Generated rule for Chiller 6 as specified.

Question: Analyze the time-series data in file xxxxx.json and look for anomalies in Chiller 6 Return Temperature.
Thought 1: I need to perform anomaly detection on Chiller 6 Return Temperature.
Action 1: RuleLogicAgent
Action Input 1: Analyze time-series data in file xxxxx.json for anomalies in Chiller 6 Return Temperature.
Observation 1: Completed anomaly detection analysis for Chiller 6 Return Temperature.

Question: Set an alert if Chiller 5 condenser pressure exceeds 250 psi for more than 10 minutes.
Thought 1: I need to create a rule to monitor Chiller 5 condenser pressure.
Action 1: RuleLogicAgent
Action Input 1: Set an alert if Chiller 5 condenser pressure exceeds 250 psi for more than 10 minutes.
Observation 1: Created rule to monitor Chiller 5 condenser pressure.

Question: Generate a rule to trigger an alert if the cooling tower fan motor current drops below 3 Amps while the fan is running.
Thought 1: I need to create a rule based on fan motor current and operational status.
Action 1: RuleLogicAgent
Action Input 1: Create a rule for cooling tower fan motor to trigger an alert if running and motor current drops below 3 Amps.
Observation 1: Rule generated for cooling tower fan motor monitoring.

Question: Perform an analysis on file cooling_tower_data.json and identify if there are any abnormal trends in fan motor current.
Thought 1: I need to detect anomalies in fan motor current data.
Action 1: RuleLogicAgent
Action Input 1: Analyze cooling_tower_data.json for abnormal trends in fan motor current.
Observation 1: Completed abnormal trend detection for fan motor current.""",
}

# ----- adapted from the paper's -----

IOT_RIG = [
    """Question: what sites are there
Thought 1: I need to get the list of available sites to answer the question.
Action 1: sites
Action Input 1: {}
Observation 1: {"sites": "MAIN"}
Thought 2: I now have the list of available sites
Action 2: Finish
Action Input 2: The sites are: "MAIN\"""",
    """Question: what assets are at site MAIN
Thought 1: I need to get the list of assets at site MAIN to answer the question.
Action 1: assets
Action Input 1: site_name=MAIN
Observation 1: Available assets: found the following 483 assets, listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.
Thought 2: I now have the list of assets for site MAIN, which is stored in a file.
Action 2: Finish
Action Input 2: The assets at site MAIN are listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.""",
    """Question: download sensor data for Chiller 4 at MAIN site
Thought 1: I need to get the sensor data for Chiller 4 at site MAIN to answer the question.
Action 1: installed_sensors
Action Input 1: asset_id=Chiller 4, site_name=MAIN
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "total_sensors": 2, "sensors": ["Chiller 4 Supply Temperature", "Chiller 4 Return Temperature"], "message": "found 2 sensors for asset_id Chiller 4 and site_name MAIN"}
Thought 2: I now have the sensor data for Chiller 4 at site MAIN.
Action 2: Finish
Action Input 2: The sensor data for Chiller 4 at site MAIN are Chiller 4 Supply Temperature and Chiller 4 Return Temperature.""",
    """Question: download asset history for Chiller 4 at MAIN site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00
Thought 1: I need to get the history for Chiller 4 at MAIN site within the specified time range.
Action 1: history
Action input 1: site_name=MAIN, asset_id=Chiller 4, start=2025-03-26T00:00:00, end=2025-04-02T00:00:00
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "observations": [{"asset_id": "Chiller 4", "timestamp": "2025-03-26T00:00:00", "Chiller 4 Supply Temperature": 6.8, "Chiller 4 Return Temperature": 12.1}, {"asset_id": "Chiller 4", "timestamp": "2025-03-26T00:15:00", "Chiller 4 Supply Temperature": 6.9, "Chiller 4 Return Temperature": 12.3}], "returned": 2, "start": "2025-03-26T00:00:00", "end": "2025-04-02T00:00:00", "message": "returned 2 observation(s) for asset_id Chiller 4; has_more=False."}
Thought 2: I now have the history for asset Chiller 4 at MAIN site within the specified time range.
Action 2: Finish
Action Input 2: The asset history for Chiller 4 at MAIN site from 2025-03-26T00:00:00 to 2025-04-02T00:00:00 has 2 observations. At 2025-03-26T00:00:00 the Chiller 4 Supply Temperature is 6.8 and the Chiller 4 Return Temperature is 12.1. At 2025-03-26T00:15:00 they are 6.9 and 12.3.""",
    """Question: what is the newest reading of Chiller 4 at MAIN site
Thought 1: I need the newest telemetry record of Chiller 4 at site MAIN.
Action 1: latest_reading
Action Input 1: site_name=MAIN, asset_id=Chiller 4
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "timestamp": "2025-04-02T00:00:00", "values": {"Chiller 4 Supply Temperature": 6.8, "Chiller 4 Return Temperature": 12.1}, "message": "latest reading for asset_id Chiller 4 at 2025-04-02T00:00:00."}
Thought 2: I now have the newest reading of Chiller 4.
Action 2: Finish
Action Input 2: The newest reading of Chiller 4 at MAIN is from 2025-04-02T00:00:00: Chiller 4 Supply Temperature 6.8 and Chiller 4 Return Temperature 12.1.""",
    """Question: describe asset Chiller 4 at MAIN site
Thought 1: I need the registry details of Chiller 4 at site MAIN.
Action 1: asset_detail
Action Input 1: site_name=MAIN, asset_id=Chiller 4
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "description": "Centrifugal chiller of the main plant", "assettype": "CHILLER", "location": "MAIN-PLANT", "n_installed_sensors": 2, "message": "asset Chiller 4 is a CHILLER with 2 installed sensors."}
Thought 2: I now have the description of Chiller 4.
Action 2: Finish
Action Input 2: Chiller 4 at MAIN is a CHILLER, described as Centrifugal chiller of the main plant, at location MAIN-PLANT, with 2 installed sensors.""",
    """Question: which assets at MAIN site have a Supply Temperature sensor
Thought 1: I need the assets of site MAIN whose sensor names contain Supply Temperature.
Action 1: find_assets_by_sensors
Action Input 1: {"site_name": "MAIN", "sensors": ["Supply Temperature"], "substring": true}
Observation 1: {"site_name": "MAIN", "query_sensors": ["Supply Temperature"], "total_assets": 1, "assets": [{"asset_id": "Chiller 4", "matched_sensors": ["Chiller 4 Supply Temperature"]}], "message": "1 asset(s) at MAIN match ['Supply Temperature'] (match=all, substring=True, source=measured)."}
Thought 2: I now have the assets with a Supply Temperature sensor.
Action 2: Finish
Action Input 2: 1 asset at MAIN has a Supply Temperature sensor: Chiller 4, with the sensor Chiller 4 Supply Temperature.""",
    """Question: how many days of telemetry does Chiller 4 at MAIN site have?
Thought 1: I should first obtain the first and the last timestamp of the telemetry of Chiller 4.
Action 1: stream_extent
Action Input 1: site_name=MAIN, asset_id=Chiller 4
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "start_time": "2025-03-26T00:00:00", "end_time": "2025-04-02T00:00:00", "total_records": 673, "message": "673 record(s) for asset_id Chiller 4 from 2025-03-26T00:00:00 to 2025-04-02T00:00:00."}
Thought 2: Now I have the first and the last timestamp and I should compute the days between them.
Action 2: Self-Ask
Action Input 2: how many days lie between 2025-03-26T00:00:00 and 2025-04-02T00:00:00
Observation 2: 7 days
Thought 3: Now I have the number of days.
Action 3: Finish
Action Input 3: The telemetry of Chiller 4 at MAIN spans 7 days, from 2025-03-26T00:00:00 to 2025-04-02T00:00:00, in 673 records.""",
]

IOT_REPRO = [
    """Question: what sites are there
Thought 1: I need to get the list of available sites to answer the question.
Action 1: sites
Action Input 1: {}
Observation 1: {"sites": "MAIN"}
Thought 2: I now have the list of available sites
Action 2: Finish
Action Input 2: The sites are: "MAIN\"""",
    """Question: what assets are at site MAIN
Thought 1: I need to get the list of assets at site MAIN to answer the question.
Action 1: assets
Action Input 1: site_name=MAIN
Observation 1: Available assets: found the following 483 assets, listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.
Thought 2: I now have the list of assets for site MAIN, which is stored in a file.
Action 2: Finish
Action Input 2: The assets at site MAIN are listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/9e17f07b-0b4a-43af-85c1-562dc21f105e.""",
    """Question: download sensor data for Chiller 4 at MAIN site
Thought 1: I need to get the sensor data for Chiller 4 at site MAIN to answer the question.
Action 1: sensors
Action Input 1: asset_id=Chiller 4, site_name=MAIN
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "total_sensors": 2, "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json", "message": "found 2 sensors for asset_id Chiller 4 and site_name MAIN. file_path contains a JSON array of Sensor data"}
Thought 2: I now have the sensor data for Chiller 4 at site MAIN, which is stored in a file.
Action 2: Finish
Action Input 2: The sensor data for Chiller 4 at site MAIN has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/c6571941-4857-4701-bd8a-9a28fa2435c3.json.""",
    """Question: download asset history for Chiller 4 at MAIN site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00
Thought 1: I need to get the history for Chiller 4 at MAIN site within the specified time range.
Action 1: history
Action input 1: site_name=MAIN, asset_id=Chiller 4, start=2025-03-26T00:00:00.000000+00:00, final=2025-04-02T00:00:00.000000+00:00
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 4", "total_observations": 3, "start": "2025-03-26T00:00:00.000000+00:00", "final": "2025-04-02T00:00:00.000000+00:00", "file_path": "/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/44934c53-0ad8-4551-adbe-5b11ad03865a.json", "message": "found 3 observations. file_path contains a JSON array of Observation data"}
Thought 2: I now have the history for asset Chiller 4 at MAIN site within the specified time range.
Action 2: Finish
Action Input 2: The asset history for Chiller 4 at MAIN site from 2025-03-26T00:00:00.000000+00:00 to 2025-04-02T00:00:00.000000+00:00 has been downloaded and is listed in file /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/44934c53-0ad8-4551-adbe-5b11ad03865a.json.""",
    """Question: what time is it in English words
Thought 1: I need the current time in English text.
Action 1: current_time_english
Action Input 1: {}
Observation 1: {"english": "2025-01-08 19:40:04", "iso": "2025-01-08T19:40:04Z"}
Thought 2: I now have the current time in English text.
Action 2: Finish
Action Input 2: The current time is 2025-01-08 19:40:04.""",
    """Question: how do I learn the correct name for the sensor Condenser Return Temperature on Chiller 1 at site MAIN?
Thought 1: I should first list all the sensors on the asset
Action 1: sensors
Action Input 1: asset_id=Chiller 1, site_name=MAIN
Observation 1: {"site_name": "MAIN", "asset_id": "Chiller 1", "total_sensors": 2, "sensors": ["Chiller 1 Condenser Water Return To Tower Temperature", "Chiller 1 Chiller Efficiency"], "message": "found 2 sensors for asset_id Chiller 1 and site_name MAIN"}
Thought 2: I should look at the sensor names and pick the closest match to Condenser Return Temperature. I should use exactly the sensor name that the list gives.
Action 2: Finish
Action Input 2: the correct sensor name is Chiller 1 Condenser Water Return To Tower Temperature""",
    """Question: I need to read the JSON file 0001.json.
Thought 1: I need to read the contents of the JSON file to access its data.
Action 1: json_reader
Action Input 1: file_name=0001.json
Observation 1: The contents of the JSON file have been read.
Thought 2: I now have the data from the JSON file.
Action 2: Finish
Action Input 2: The contents of 0001.json have been successfully read.""",
    """Question: how do I calculate the start date for last week or past week?
Thought 1: I should first obtain the current date and then subtract 7 days from the current date.
Action 1: current_date_time
Action Input 1: {}
Observation 1: {"currentDateTime": "2025-01-08T19:40:04.664816+00:00", "currentDateTimeDescription": "Today's date is 2025-01-08 and time is 19:40:04."}
Thought 2: Now, I have the current date and I should proceed to generate the start date for last week.
Action 2: Self-Ask
Action Input 2: what is the start date for last week if current date is 2025-01-08T19:40:04.664816+00:00
Observation 2: based on current date, the last week starts on 2025-01-01T00:00:00.000000+00:00
Thought 3: Now I have the start date for last week
Action 3: Finish
Action Input 3: the start date for last week is 2025-01-01T00:00:00.000000+00:00.""",
]

TSFM_REPRO = [
    """Question: What types of analysis are supported?
Thought 1: I need to query available AI Tasks
Action 1: get_ai_tasks
Action Input 1: {}
Observation 1: [('tsfm-ad','Anomaly detection'), ('tsfm-ad','Multivariate Forecasting')]
Thought 2: I now have the list of AI Tasks
Action 2: Finish
Action Input 2: the available AI tasks are [('tsfm-ad','Anomaly detection'), ('tsfm-ad','Multivariate Forecasting')]""",
    """Question: What pretrained models are available?
Thought 1: I need to query available pretrained models.
Action 1: get_tsfm_models
Action Input 1: {}
Observation 1: [('ttm_96_28', 'model checkpoint', 'description'),('ttm_512_96', 'model checkpoint', 'description')]
Thought 2: I now have the available pretrained models.
Action 2: Finish
Action Input 2: The available pretrained models are [('ttm_96_28', 'model checkpoint', 'description'),('ttm_512_96', 'model checkpoint', 'description')]""",
    """Question: Provide a model checkpoint for the forecasting task.
Thought 1: I need to query available pretrained models.
Action 1: get_tsfm_models
Action Input 1: {}
Observation 1: [('ttm_96_28', 'model checkpoint', 'description'),('ttm_512_96', 'model checkpoint', 'description')]
Thought 2: I now have the available pretrained models.
Action 2: Finish
Action Input 2: The available pretrained models are [('ttm_96_28', 'model checkpoint', 'description'),('ttm_512_96', 'model checkpoint', 'description')]""",
    """Question: Forecast 'Air Handler AIX765 Condenser Water Flow' using data in 'data/tsfm_test_data/chiller9_annotated_small_test.csv'. Use parameter 'Timestamp' as a timestamp.
Thought 1: I need to perform Time Series Forecasting of 'Air Handler AIX765 Condenser Water Flow' using data in data/tsfm_test_data/chiller9_annotated_small_test.csv
Action 1: run_tsfm_forecasting
Action Input 1: dataset_path=data/tsfm_test_data/chiller9_annotated_small_test.csv, target_columns=Air Handler AIX765 Condenser Water Flow, timestamp_column=Timestamp
Observation 1: {%JSON_RESULT}
Thought 2: I now have Forecasting results of 'Air Handler AIX765 Condenser Water Flow' using data in data/tsfm_test_data/chiller9_annotated_small_test.csv
Action 2: Finish
Action Input 2: The Forecasting results of 'Air Handler AIX765 Condenser Water Flow' using data in data/tsfm_test_data/chiller9_annotated_small_test.csv are {%JSON_RESULT}""",
    """Question: For Air Handler AIX765 finetune a forecasting model of 'Air Handler AIX765 Condenser Water Flow' using data in 'data/tsfm_test_data/chiller9_finetuning.csv'. 
Thought 1: I need to finetune a new Time Series Forecasting model of 'Air Handler AIX765 Condenser Water Flow' using data in data/tsfm_test_data/chiller9_finetuning.csv
Action 1: run_tsfm_finetuning
Action Input 1: n_finetune=0.05, dataset_path=data/tsfm_test_data/chiller9_finetuning.csv, target_columns=Air Handler AIX765 Condenser Water Flow
Observation 1: {%JSON_RESULT}
Thought 2: I now have a finetuned model for forecasting of 'Air Handler AIX765 Condenser Water Flow' using data in data/tsfm_test_data/chiller9_finetuning.csv. The model is saved in save_model_dir=tunedmodels and the results are stored in results_file.
Action 2: Finish
Action Input 2: The finetuning of the forecasting model of 'Air Handler AIX765 Condenser Water Flow' is done, and its results are {%JSON_RESULT}""",
    """Question: Compute anomaly detection for 'Chiller 6 Return Temperature' using data in '/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json'. Use parameter 'Timestamp' as a timestamp. 
Thought 1: I need to perform Time Series anomaly detection of 'Chiller 6 Return Temperature' using data in /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json
Action Input 1: dataset_path=/var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json, timestamp_column=Timestamp, target_columns=Chiller 6 Return Temperature
Observation 1: {%CSV_RESULT}
Thought 2: I now have anomaly detection results of 'Chiller 6 Return Temperature' using data in /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json
Action 2: Finish
Action Input 2: The anomaly detection results of 'Chiller 6 Return Temperature' using data in /var/folders/fz/l1h7gpv96rv5lg6m_d6bk0gc0000gn/T/cbmdir/d0237195-1716-487a-ae15-f9fd4f3ac8ea.json are {%CSV_RESULT}""",
]

# ----- written for this port -----

TASK_SUPERVISOR = """Question: What are the budget and the horizon of the task on asset CHL-A100?
Thought 1: I need the asset profile, which states the budget and the horizon.
Action 1: TaskAgent
Action Input 1: request=read the asset profile of CHL-A100
Observation 1: The asset profile of CHL-A100, a CHILLER at site SiteX: the components are compressor, condenser, evaporator and fan, the budget is 100 and the horizon is 40.

Question: Record the diagnosis compressor 2, condenser 1, evaporator 1, fan 3 with the evidence obs/2025-04-02T00:15:00/Supply Temperature and obs/2025-04-02T00:30:00/Supply Temperature.
Thought 1: I need to submit the diagnosis the question gives.
Action 1: TaskAgent
Action Input 1: request=submit the diagnosis compressor 2, condenser 1, evaporator 1, fan 3 with the evidence obs/2025-04-02T00:15:00/Supply Temperature and obs/2025-04-02T00:30:00/Supply Temperature
Observation 1: The diagnosis compressor 2, condenser 1, evaporator 1, fan 3 is submitted with the evidence obs/2025-04-02T00:15:00/Supply Temperature and obs/2025-04-02T00:30:00/Supply Temperature.

Question: End the task with the sites of the registry as its summary.
Thought 1: I need the list of the sites from IoTAgent first.
Action 1: IoTAgent
Action Input 1: request=list all the sites
Observation 1: {"sites": ["SiteX", "SiteY", "SiteZ"], "message": "List of available sites retrieved."}
Thought 2: The sites are SiteX, SiteY and SiteZ. I need to end the task with them as its summary.
Action 2: TaskAgent
Action Input 2: request=end the task with the summary the sites are SiteX, SiteY and SiteZ
Observation 2: The task is ended with the summary the sites are SiteX, SiteY and SiteZ."""

CAUSAL_SUPERVISOR = """Question: Which evidence about CHL-A100 has been released so far?
Thought 1: I need the released evidence of every sensor group from CausalDiagnosisAgent.
Action 1: CausalDiagnosisAgent
Action Input 1: request=list the released evidence of every sensor group
Observation 1: The released evidence of every sensor group is 2 items from 1 observation at cycle 3: obs/1/Supply Temperature_mean is 7.9, outside its normal range of 5.5 to 7.5, and obs/1/Return Temperature_mean is 12.4, within its normal range of 11.0 to 13.0.

Question: Rank the candidate root causes of the readings of CHL-A100.
Thought 1: I need the ranking of the candidate root causes from CausalDiagnosisAgent.
Action 1: CausalDiagnosisAgent
Action Input 1: request=rank the three most likely root causes
Observation 1: The three most likely root causes are, by rank: compressor 2, condenser 0, evaporator 0, fan 0 with probability 0.62, then compressor 1, condenser 0, evaporator 0, fan 0 with probability 0.21, then compressor 2, condenser 0, evaporator 0, fan 3 with probability 0.09. The ranking does not say read first.

Question: Estimate the effect of replacing the compressor of CHL-A100.
Thought 1: I need the predicted effect of the replacement from CausalDiagnosisAgent.
Action 1: CausalDiagnosisAgent
Action Input 1: request=estimate the effect of the action replace:compressor
Observation 1: The predicted effect of replacing the compressor is prediction pred/1: the action addresses a fault with probability 0.95, the predicted recovery is 0.81, the price is 35 and the duration is 6 cycles.

Question: Check whether the repair of CHL-A100 did what prediction pred/1 said.
Thought 1: I need the verification of prediction pred/1 from CausalDiagnosisAgent.
Action 1: CausalDiagnosisAgent
Action Input 1: request=verify the outcome of prediction pred/1
Observation 1: Prediction pred/1 of the action replace:compressor is verified against the reading after the repair: the outcome is consistent with the prediction, the belief after the reading confirms it against its strongest rival and the recovery posterior is 0.97.

Question: How likely is each candidate root cause of CHL-A100 after the last reading?
Thought 1: I need a new ranking of the candidate root causes from CausalDiagnosisAgent.
Action 1: CausalDiagnosisAgent
Action Input 1: request=rank the candidate root causes
Observation 1: The candidate root causes are, by rank: compressor 2, condenser 0, evaporator 0, fan 0 with probability 0.91, then compressor 1, condenser 0, evaporator 0, fan 0 with probability 0.06."""

FMSR_RIG = [
    """Question: List all failure modes of asset CHL-A100.
Thought 1: I need the failure modes of the asset class of CHL-A100, which is a chiller.
Action 1: get_failure_modes
Action Input 1: asset_class=chiller
Observation 1: {"asset_class": "chiller", "failure_modes": ["Compressor Overheating", "Evaporator Fouling", "Low Refrigerant Charge", "Sensor Faults"]}
Thought 2: I now have the failure modes of CHL-A100.
Action 2: Finish
Action Input 2: The failure modes of the chiller class, the class of CHL-A100, are Compressor Overheating, Evaporator Fouling, Low Refrigerant Charge and Sensor Faults.""",
]

FMSR_REPRO = [
    """Question: List all failure modes of asset CHL-A100.
Thought 1: I need the failure modes of the asset CHL-A100.
Action 1: get_failure_modes
Action Input 1: asset_name=CHL-A100
Observation 1: {"asset_name": "CHL-A100", "failure_modes": ["Compressor Overheating", "Evaporator Fouling", "Low Refrigerant Charge", "Sensor Faults"]}
Thought 2: I now have the failure modes of CHL-A100.
Action 2: Finish
Action Input 2: The failure modes of CHL-A100 are Compressor Overheating, Evaporator Fouling, Low Refrigerant Charge and Sensor Faults.""",
]

WO_RIG = [
    """Question: List the open work orders of site SiteX.
Thought 1: I need the work orders of site SiteX whose status is open.
Action 1: list_workorders
Action Input 1: site_id=SiteX, status=OPEN
Observation 1: {"site_id": "SiteX", "status": "OPEN", "total": 1, "work_orders": [{"wonum": "WO-12345", "description": "inspect the compressor", "assetnum": "CHL-A100_compressor", "status": "APPR", "worktype": "CM", "wopriority": 2}], "message": "Found 1 work order(s)."}
Thought 2: I now have the open work orders of SiteX.
Action 2: Finish
Action Input 2: SiteX has 1 open work order: WO-12345 on CHL-A100_compressor, inspect the compressor, of work type CM with priority 2 and status APPR.""",
    """Question: Show work order WO-12345 of site SiteX.
Thought 1: I need the details of work order WO-12345 at SiteX.
Action 1: get_workorder
Action Input 1: wonum=WO-12345, site_id=SiteX
Observation 1: {"work_order": {"wonum": "WO-12345", "description": "inspect the compressor", "siteid": "SiteX", "assetnum": "CHL-A100_compressor", "status": "APPR", "worktype": "CM", "wopriority": 2, "schedstart": "2025-04-02T00:00:00+00:00", "schedfinish": "2025-04-02T01:30:00+00:00"}, "message": "Work order WO-12345 at SiteX."}
Thought 2: I now have the details of WO-12345.
Action 2: Finish
Action Input 2: Work order WO-12345 at SiteX is a CM order on CHL-A100_compressor, inspect the compressor, with priority 2 and status APPR, scheduled from 2025-04-02T00:00:00+00:00 to 2025-04-02T01:30:00+00:00.""",
    """Question: What does failure code FC001 mean?
Thought 1: I need the failure code catalog entry of FC001.
Action 1: get_failure_codes
Action Input 1: code=FC001
Observation 1: {"code": "FC001", "total": 1, "failure_codes": [{"code": "FC001", "description": "equipment does not start"}], "message": "Found 1 failure code(s) for FC001."}
Thought 2: I now have the meaning of FC001.
Action 2: Finish
Action Input 2: Failure code FC001 means equipment does not start.""",
    """Question: Create a corrective work order to inspect the compressor of CHL-A100, child asset CHL-A100_compressor, at site SiteX.
Thought 1: I need to create a corrective work order on the child asset CHL-A100_compressor at SiteX.
Action 1: generate_work_order
Action Input 1: description=inspect the compressor, asset_num=CHL-A100_compressor, site_id=SiteX, priority=2, work_type=CM
Observation 1: {"wonum": "WO-12345", "siteid": "SiteX", "status": "WAPPR", "work_order": {"wonum": "WO-12345", "description": "inspect the compressor", "siteid": "SiteX", "assetnum": "CHL-A100_compressor", "status": "WAPPR", "worktype": "CM", "wopriority": 2}, "message": "Work order WO-12345 created (WAPPR)."}
Thought 2: The work order is created.
Action 2: Finish
Action Input 2: The corrective work order WO-12345 to inspect the compressor of CHL-A100 is created on CHL-A100_compressor at SiteX with status WAPPR.""",
    """Question: What are the work order KPIs of site SiteX over the last 12 months?
Thought 1: I need the work order KPIs of SiteX for 12 months.
Action 1: get_workorder_kpis
Action Input 1: site_id=SiteX, period_months=12
Observation 1: {"site_id": "SiteX", "period_months": 12, "total_workorders": 14, "backlog": 3, "overdue": 1, "avg_completion_hrs": 6.5, "message": "KPIs for SiteX over 12 month(s)."}
Thought 2: I now have the KPIs of SiteX.
Action 2: Finish
Action Input 2: Over 12 months SiteX has 14 work orders, a backlog of 3, 1 overdue work order and an average completion time of 6.5 hours.""",
]

WO_REPRO = [
    """Question: Get the work orders of equipment CU02013 for year 2017.
Thought 1: I need the work orders of CU02013 between 2017-01-01 and 2017-12-31.
Action 1: get_work_orders
Action Input 1: equipment_id=CU02013, start_date=2017-01-01, end_date=2017-12-31
Observation 1: {"equipment_id": "CU02013", "total": 3, "work_orders": [{"wo_id": "WO-1001", "wo_description": "Inspect compressor"}, ...]}
Thought 2: I now have the work orders of CU02013 for 2017.
Action 2: Finish
Action Input 2: CU02013 had 3 work orders in 2017, among them WO-1001 to inspect the compressor.""",
    """Question: List the preventive work orders of CU02013 in 2017.
Thought 1: I need the preventive work orders of CU02013 in 2017.
Action 1: get_preventive_work_orders
Action Input 1: equipment_id=CU02013, start_date=2017-01-01, end_date=2017-12-31
Observation 1: {"equipment_id": "CU02013", "total": 2, "work_orders": [...]}
Thought 2: I now have the preventive work orders.
Action 2: Finish
Action Input 2: CU02013 had 2 preventive work orders in 2017.""",
    """Question: Which failure codes exist?
Thought 1: I need the list of failure codes.
Action 1: get_failure_codes
Action Input 1: {}
Observation 1: {"total": 2, "failure_codes": [{"category": "Mechanical", "primary_code": "FC001", "primary_code_description": "Bearing wear"}, ...]}
Thought 2: I now have the failure codes.
Action 2: Finish
Action Input 2: The failure codes include FC001, a mechanical failure for bearing wear.""",
    """Question: Show the events of CU02013 in June 2020.
Thought 1: I need the work orders, alerts and anomalies of CU02013 in June 2020.
Action 1: get_events
Action Input 1: equipment_id=CU02013, start_date=2020-06-01, end_date=2020-06-30
Observation 1: {"equipment_id": "CU02013", "total": 4, "events": [...]}
Thought 2: I now have the events of CU02013 for June 2020.
Action 2: Finish
Action Input 2: CU02013 had 4 events in June 2020.""",
    """Question: Predict the next work order type of CU02013.
Thought 1: I need the prediction of the next work order type from the history of CU02013.
Action 1: predict_next_work_order
Action Input 1: equipment_id=CU02013
Observation 1: {"equipment_id": "CU02013", "predictions": [{"category": "Corrective", "probability": 0.6}, ...]}
Thought 2: I now have the prediction.
Action 2: Finish
Action Input 2: The next work order of CU02013 is most likely corrective.""",
]

DIAGNOSIS_TOOL = "submit_diagnosis"  # the task tool whose example is written at run time for the asset of the run
# the example of DIAGNOSIS_TOOL with three fields: the named levels of the question and of the Finish line, the arguments of the
# Action Input and the diagnosis of the observation
DIAGNOSIS_TEMPLATE = """Question: submit the diagnosis <named> with the evidence obs/2025-04-02T00:15:00/Supply Temperature and obs/2025-04-02T00:30:00/Supply Temperature
Thought 1: I need to submit the diagnosis the question gives.
Action 1: submit_diagnosis
Action Input 1: <arguments>, evidence_ids=obs/2025-04-02T00:15:00/Supply Temperature,obs/2025-04-02T00:30:00/Supply Temperature
Observation 1: {"diagnosis": <diagnosis>, "evidence_ids": ["obs/2025-04-02T00:15:00/Supply Temperature", "obs/2025-04-02T00:30:00/Supply Temperature"]}
Thought 2: The diagnosis is recorded.
Action 2: Finish
Action Input 2: The diagnosis <named> is submitted with the evidence obs/2025-04-02T00:15:00/Supply Temperature and obs/2025-04-02T00:30:00/Supply Temperature."""


def diagnosis_example(parameters, start):
    """The example of DIAGNOSIS_TOOL for one asset. parameters are the names of the tool's parameters in the order of its
    listed schema, and start is the registered example start of the run, {component: level}, a development start. The
    components shown are the parameters that the start names, in the order of the schema, and evidence_ids takes the two
    identifiers of the template. Every key of the start must be a parameter, and every parameter outside the start must be
    evidence_ids."""
    if start is None:
        raise ValueError("the diagnosis example needs the registered example start of the run, which the driver computes")
    shown = [p for p in parameters if p in start]
    extra = sorted(set(start) - set(parameters))
    missing = [p for p in parameters if p not in start and p != "evidence_ids"]
    if extra or missing:
        raise ValueError(f"the example start does not fit the parameters of {DIAGNOSIS_TOOL}: it names {extra}, which the tool "
                         f"does not have, and leaves out {missing}")
    named = ", ".join(f"{c} {start[c]}" for c in shown)
    arguments = ", ".join(f"{c}={start[c]}" for c in shown)
    return (DIAGNOSIS_TEMPLATE.replace("<named>", named).replace("<arguments>", arguments)
            .replace("<diagnosis>", json.dumps({c: start[c] for c in shown})))


TASK_TOOLS = {
    'get_asset_profile': """Question: read the asset profile
Thought 1: I need the asset profile to answer the question.
Action 1: get_asset_profile
Action Input 1: {}
Observation 1: {"asset_id": "CHL-A100", "asset": {"asset_id": "CHL-A100", "asset_type": "CHILLER", "site": "SiteX", "children": {"compressor": "CHL-A100_compressor", "condenser": "CHL-A100_condenser", "evaporator": "CHL-A100_evaporator", "fan": "CHL-A100_fan"}}, "components": ["compressor", "condenser", "evaporator", "fan"], "sensor_groups": {"tier1": ["Supply Temperature"], "tier2": ["Supply Temperature", "Return Temperature"]}, "budget": 100, "horizon": 40}
Thought 2: I now have the asset profile.
Action 2: Finish
Action Input 2: The asset profile of CHL-A100, a CHILLER at site SiteX: the components are compressor, condenser, evaporator and fan, with the child assets CHL-A100_compressor, CHL-A100_condenser, CHL-A100_evaporator and CHL-A100_fan. The sensor groups are tier1 with Supply Temperature and tier2 with Supply Temperature and Return Temperature. The budget is 100 and the horizon is 40.""",
    'get_current_time': """Question: what time is it on the asset's clock
Thought 1: I need the current time of the asset's clock.
Action 1: get_current_time
Action Input 1: {}
Observation 1: {"time": 3, "horizon": 40, "seconds_per_cycle": 900, "current_datetime": "2025-04-02T00:45:00"}
Thought 2: I now have the time.
Action 2: Finish
Action Input 2: The asset's clock is at cycle 3 of a horizon of 40, a cycle is 900 seconds long, and the date and time of the current cycle is 2025-04-02T00:45:00.""",
    'wait_for_notification': """Question: wait up to 5 cycles for the next notification
Thought 1: I need to wait for at most 5 cycles.
Action 1: wait_for_notification
Action Input 1: timeout=5
Observation 1: {"time": 8, "waited": 5, "ended_by": "timeout"}
Thought 2: The wait is over.
Action 2: Finish
Action Input 2: I waited 5 cycles, the clock is now at cycle 8, and the wait ended by timeout.""",
    'lookup_fix': """Question: look up the standard fix of the compressor
Thought 1: I need the repair manual of the compressor.
Action 1: lookup_fix
Action Input 1: component=compressor
Observation 1: {"component": "compressor", "standard": {"mode": "standard", "fix": "replace", "part": 25, "repair_charge": 10, "price": 35, "lead_time": 2, "repair_downtime": 4}, "priority": 1, "dependencies": [], "effect": "At the close of the order the compressor returns to its healthy level 0."}
Thought 2: I now have the repair manual of the compressor.
Action 2: Finish
Action Input 2: The standard fix of the compressor is replace, at a price of 35 (part 25 plus repair charge 10), with a lead time of 2 cycles and a repair downtime of 4 cycles. Its priority is 1 and it has no dependencies. At the close of the order the compressor returns to its healthy level 0.""",
    DIAGNOSIS_TOOL: DIAGNOSIS_TEMPLATE,
    'declare_insufficient_evidence': """Question: declare that the evidence is insufficient and that 2 cycles of the sensor group tier2 would separate the candidates, citing obs/2025-04-02T00:15:00/Supply Temperature
Thought 1: I need to declare the evidence insufficient with the reading the question names.
Action 1: declare_insufficient_evidence
Action Input 1: sensor_group=tier2, cycles=2, evidence_ids=obs/2025-04-02T00:15:00/Supply Temperature
Observation 1: {"separating_test": {"sensor_group": "tier2", "cycles": 2}, "evidence_ids": ["obs/2025-04-02T00:15:00/Supply Temperature"]}
Thought 2: The declaration is recorded.
Action 2: Finish
Action Input 2: The evidence is declared insufficient, naming 2 cycles of tier2, with the evidence obs/2025-04-02T00:15:00/Supply Temperature.""",
    'end_episode': """Question: end the task with the summary inspection complete
Thought 1: I need to end the task with the summary the question gives.
Action 1: end_episode
Action Input 1: summary=inspection complete
Observation 1: {"summary": "inspection complete"}
Thought 2: The task is ended.
Action 2: Finish
Action Input 2: The task is ended with the summary inspection complete.""",
}

CAUSAL_TOOLS = {
    'get_episode_evidence': """Question: list the evidence released so far for every sensor group
Thought 1: I need the released evidence of all sensor groups.
Action 1: get_episode_evidence
Action Input 1: sensor_group=all
Observation 1: {"sensor_group": "all", "n_observations": 1, "items": [{"evidence_id": "obs/1/Supply Temperature_mean", "field": "Supply Temperature_mean", "value": 7.9, "t": 3, "normal_low": 5.5, "normal_high": 7.5, "in_normal_range": false}, {"evidence_id": "obs/1/Return Temperature_mean", "field": "Return Temperature_mean", "value": 12.4, "t": 3, "normal_low": 11.0, "normal_high": 13.0, "in_normal_range": true}], "evidence_ids": ["obs/1/Supply Temperature_mean", "obs/1/Return Temperature_mean"]}
Thought 2: I now have the released evidence.
Action 2: Finish
Action Input 2: The released evidence of every sensor group is 2 items from 1 observation at cycle 3: obs/1/Supply Temperature_mean is 7.9, outside its normal range of 5.5 to 7.5, and obs/1/Return Temperature_mean is 12.4, within its normal range of 11.0 to 13.0.""",
    'rank_root_causes': """Question: rank the three most likely root causes
Thought 1: I need the ranking of the three most likely root causes.
Action 1: rank_root_causes
Action Input 1: top_k=3
Observation 1: {"candidates": [{"rank": 1, "combination": {"compressor": 2, "condenser": 0, "evaporator": 0, "fan": 0}, "probability": 0.62}, {"rank": 2, "combination": {"compressor": 1, "condenser": 0, "evaporator": 0, "fan": 0}, "probability": 0.21}, {"rank": 3, "combination": {"compressor": 2, "condenser": 0, "evaporator": 0, "fan": 3}, "probability": 0.09}], "current_fault_probability": {"compressor": 0.93, "condenser": 0.02, "evaporator": 0.01, "fan": 0.11}, "read_first": false, "evidence_ids": ["obs/1/Supply Temperature_mean", "obs/1/Return Temperature_mean"]}
Thought 2: I now have the ranking.
Action 2: Finish
Action Input 2: The three most likely root causes are, by rank: compressor 2, condenser 0, evaporator 0, fan 0 with probability 0.62, then compressor 1, condenser 0, evaporator 0, fan 0 with probability 0.21, then compressor 2, condenser 0, evaporator 0, fan 3 with probability 0.09. The ranking does not say read first.""",
    'estimate_intervention_effect': """Question: estimate the effect of the action replace:compressor
Thought 1: I need the predicted effect of replacing the compressor.
Action 1: estimate_intervention_effect
Action Input 1: action=replace:compressor
Observation 1: {"prediction_id": "pred/1", "action": "replace:compressor", "addresses_fault": 0.95, "recovery": 0.81, "price": 35, "duration": 6}
Thought 2: I now have the prediction.
Action 2: Finish
Action Input 2: The predicted effect of replacing the compressor is prediction pred/1: the action addresses a fault with probability 0.95, the predicted recovery is 0.81, the price is 35 and the duration is 6 cycles. pred/1 is the prediction ID to cite in the work order and to pass to verify_outcome.""",
    'verify_outcome': """Question: verify the outcome of prediction pred/1
Thought 1: I need to verify prediction pred/1 against the reading after the repair.
Action 1: verify_outcome
Action Input 1: prediction_id=pred/1
Observation 1: {"prediction_id": "pred/1", "action": "replace:compressor", "status": "verified", "consistent": true, "confirmed": true, "recovery_posterior": 0.97}
Thought 2: I now have the verification.
Action 2: Finish
Action Input 2: Prediction pred/1 of the action replace:compressor is verified against the reading after the repair: the outcome is consistent with the prediction, the belief after the reading confirms it against its strongest rival and the recovery posterior is 0.97.""",
}
