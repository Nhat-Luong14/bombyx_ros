#!/usr/bin/env python3
import json
import datetime 

logging =1
"""
Write the log for process visualize and debugging
"""
def logging():
    time_info = datetime.datetime.now()
    filename = str(time_info) +".json" 

    # Create new json file with name is the time of experiment
    json_log = {"log":[]
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_log, f, ensure_ascii=False, indent=4)

    return filename


"""
Append log into json file
"""
def write_json(new_data, filename):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["log"].append(new_data)
        file.seek(0)    # Sets file's current position at offset.
        json.dump(file_data, file, indent = 4)