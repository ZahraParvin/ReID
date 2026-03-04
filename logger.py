import requests
import time
import csv

SERVER_URL = "http://131.246.34.124:5005"

def read_log_continuously():
    """Continuously read and display new log entries"""
    last_line_count = 0
    
    while True:
        try:
            response = requests.get(f"{SERVER_URL}/get_recent_logs?lines=1000", timeout=5)
            if response.status_code == 200:
                logs = response.json()['logs']
                
                # Display only new lines
                if len(logs) > last_line_count:
                    for line in logs[last_line_count:]:
                        print(line)
                    last_line_count = len(logs)
            
            time.sleep(1)  # Poll every second
            
        except Exception as e:
            print(f"Error reading logs: {e}")
            time.sleep(5)

def download_full_log():
    """Download the complete log file"""
    try:
        response = requests.get(f"{SERVER_URL}/get_log", timeout=10)
        if response.status_code == 200:
            with open("downloaded_log.csv", 'wb') as f:
                f.write(response.content)
            print("Log file downloaded successfully!")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error downloading log: {e}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Monitor logs in real-time")
    print("2. Download full log file")
    choice = input("Enter choice (1/2): ")
    
    if choice == "1":
        read_log_continuously()
    elif choice == "2":
        download_full_log()
