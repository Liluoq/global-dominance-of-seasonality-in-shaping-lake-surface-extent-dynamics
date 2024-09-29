SERVICE_ACCOUNTS = [
    "liluoqi2000@ee-llq846183119.iam.gserviceaccount.com",
    "llq-104@ee-llq846183119.iam.gserviceaccount.com",
    "llq1-102@ee-llq846183119.iam.gserviceaccount.com",
    "sa1-319@ee-llq846183119.iam.gserviceaccount.com",
    "sa2-35@ee-llq846183119.iam.gserviceaccount.com",
    "sa3-827@ee-llq846183119.iam.gserviceaccount.com",
    "sa4-848@ee-llq846183119.iam.gserviceaccount.com",
    "sa5-363@ee-llq846183119.iam.gserviceaccount.com",
    "sa6-153@ee-llq846183119.iam.gserviceaccount.com",
    "sa7-840@ee-llq846183119.iam.gserviceaccount.com",
    "sa8-620@ee-llq846183119.iam.gserviceaccount.com",
    "sa9-804@ee-llq846183119.iam.gserviceaccount.com",
    "sa10-423@ee-llq846183119.iam.gserviceaccount.com",
    "sa11-836@ee-llq846183119.iam.gserviceaccount.com",
    "sa12-422@ee-llq846183119.iam.gserviceaccount.com",
    "sa13-5@ee-llq846183119.iam.gserviceaccount.com",
    "sa14-787@ee-llq846183119.iam.gserviceaccount.com",
    "sa15-195@ee-llq846183119.iam.gserviceaccount.com",
    "sa16-748@ee-llq846183119.iam.gserviceaccount.com",
    "sa17-879@ee-llq846183119.iam.gserviceaccount.com",
    "sa18-373@ee-llq846183119.iam.gserviceaccount.com",
    "sa19-993@ee-llq846183119.iam.gserviceaccount.com",
    "sa20-892@ee-llq846183119.iam.gserviceaccount.com",
]
KEYPATHS = [
    "./GoogleCloud_serviceAccounts/ee-llq846183119-a13a1c615389.json",
    "./GoogleCloud_serviceAccounts/ee-llq846183119-llq.json",
    "./GoogleCloud_serviceAccounts/ee-llq846183119-llq1.json",
    "./GoogleCloud_serviceAccounts/sa1.json",
    "./GoogleCloud_serviceAccounts/sa2.json",
    "./GoogleCloud_serviceAccounts/sa3.json",
    "./GoogleCloud_serviceAccounts/sa4.json",
    "./GoogleCloud_serviceAccounts/sa5.json",
    "./GoogleCloud_serviceAccounts/sa6.json",
    "./GoogleCloud_serviceAccounts/sa7.json",
    "./GoogleCloud_serviceAccounts/sa8.json",
    "./GoogleCloud_serviceAccounts/sa9.json",
    "./GoogleCloud_serviceAccounts/sa10.json",
    "./GoogleCloud_serviceAccounts/sa11.json",
    "./GoogleCloud_serviceAccounts/sa12.json",
    "./GoogleCloud_serviceAccounts/sa13.json",
    "./GoogleCloud_serviceAccounts/sa14.json",
    "./GoogleCloud_serviceAccounts/sa15.json",
    "./GoogleCloud_serviceAccounts/sa16.json",
    "./GoogleCloud_serviceAccounts/sa17.json",
    "./GoogleCloud_serviceAccounts/sa18.json",
    "./GoogleCloud_serviceAccounts/sa19.json",
    "./GoogleCloud_serviceAccounts/sa20.json"
]

import ee
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
import sys
import os


"""Earth Engine API based functions"""

def auth_gee_service_account(index):
    service_account = SERVICE_ACCOUNTS[index]
    keypath = KEYPATHS[index]
    credentials = ee.ServiceAccountCredentials(service_account, keypath)
    ee.Initialize(credentials)
    print(f"(GEE) Authenticated {service_account} successfully!")
    return None
    
def auth_gcs_service_account(index):
    keypath = KEYPATHS[index]
    service_account = SERVICE_ACCOUNTS[index]
    os.system(f'gcloud auth activate-service-account --key-file {keypath}')
    print(f"(GCS) Authenticated {service_account} successfully!")
    return None

def cancel_all_operations():
    operationList = ee.data.listOperations()
    cnt = 0
    for operation in operationList:
        state = operation['metadata']['state']
        if((state == 'PENDING') or (state == 'RUNNING')):
            name = operation['name']
            ee.data.cancelOperation(name)
            cnt += 1
        else:
            continue
    print(f'{cnt} task canceled')

def list_all_operation_status():
    pending = 0
    running = 0
    failed = 0
    cancelled = 0
    succeeded = 0
    operationList = ee.data.listOperations()
    for operation in operationList:
        state = operation['metadata']['state']
        if(state == 'PENDING'):
            pending += 1
        elif(state == 'RUNNING'):
            running += 1
        elif(state == 'FAILED'):
            failed += 1
        elif(state == 'CANCELLED'):
            cancelled += 1
        elif(state == 'SUCCEEDED'):
            succeeded += 1
    taskStatusList = {
        'PENDING': pending,
        'RUNNING': running,
        'FAILED': failed,
        'CANCELLED': cancelled,
        'SUCCEEDED': succeeded
    }
    return taskStatusList

def present_operations():
    taskStatusList = list_all_operation_status()
    present = taskStatusList['PENDING'] + taskStatusList['RUNNING']
    return present

def list_all_accounts_operations_present():
    accountNumber = len(SERVICE_ACCOUNTS)
    tasks = {}
    for i in range(accountNumber):
        presentAccount = SERVICE_ACCOUNTS[i]
        presentKeypath = KEYPATHS[i]
        switch_service_account(presentAccount, presentKeypath)
        presentTaskCnt = present_operations()
        tasks[presentAccount] = presentTaskCnt
    for accountId, taskCnt in tasks.items():
        print(f"{accountId}: {taskCnt} tasks present")
    return tasks

def list_all_accounts_operations_status():
    accountNumber = len(SERVICE_ACCOUNTS)
    tasksStatus = {}
    for i in range(accountNumber):
        presentAccount = SERVICE_ACCOUNTS[i]
        presentKeypath = KEYPATHS[i]
        switch_service_account(presentAccount, presentKeypath)
        presentTaskStatus = list_all_operation_status()
        tasksStatus[presentAccount] = presentTaskStatus
    for accountId, taskStatus in tasksStatus.items():
        pending = taskStatus["PENDING"]
        running = taskStatus["RUNNING"]
        failed = taskStatus["FAILED"]
        cancelled = taskStatus["CANCELLED"]
        succeeded = taskStatus["SUCCEEDED"]
        print(f"{accountId}: PENDING:{pending}, RUNNING:{running}, FAILED:{failed}, CANCELLED:{cancelled}, SUCCEEDED:{succeeded}")
    return tasksStatus

def date_range_list(start_date, end_date, gap_month):
    months = ee.Number(ee.Date(end_date).difference(ee.Date(start_date), 'month').round())
    seq_list = ee.List.sequence(0, months.subtract(1), ee.Number(gap_month))
    daterange_list = seq_list.map(lambda i: ee.DateRange(ee.Date(start_date).advance(ee.Number(i), 'month'), ee.Date(start_date).advance(ee.Number(i), 'month').advance(ee.Number(gap), 'month')))
    return daterange_list



""""Cloud Storage based functions"""

def cs_service_account_authenticate(keypath):
    credentials = service_account.Credentials.from_service_account_file(keypath)
    storage_client = storage.Client(credentials = credentials)
    print("Successfully authenticated!")
    return storage_client

def cs_get_bucket(storage_client, bucket_name):
    bucket = storage_client.get_bucket(bucket_name)
    print(f"Bucket {bucket_name} successfully fetched!")
    return bucket

def download_file_from_bucket(bucket, filename, local_folder, sa_index, 
    bucket_folder = None):

    if(sa_index != None):
        switch_service_account(SERVICE_ACCOUNTS[sa_index], KEYPATHS[sa_index])
    if(bucket_folder != None):
        bucket_filename = bucket_folder + filename
    blob = bucket.blob(bucket_filename)
    localFileName = local_folder + filename
    blob.download_to_filename(localFileName)
    print(f"{filename} successfully downloaded")
    return None

def download_files_in_folder(bucket, local_folder,
    sa_index = None, left_position = 0, right_position = 1, 
    write_to_file = None, bucket_folder = None, conti = None):

    if(sa_index != None):
        switch_service_account(SERVICE_ACCOUNTS[sa_index], KEYPATHS[sa_index])
    file_blobs = bucket.list_blobs(prefix = bucket_folder)
    filenames = [file.name for file in file_blobs]
    file_numbers = len(filenames)
    left_index = int(np.floor(left_position * file_numbers))
    right_index = int(np.floor(right_position * file_numbers))
    if(conti != None):
        if((conti >= left_index) and (conti < right_index)):
            index = conti
        else:
            raise Exception('conti is not between left_index and right_index')
    else:
        index = left_index
    while(index < right_index):
        filename = filenames[index]
        if(filename != bucket_folder):
            file_blob = bucket.blob(filename)
            local_filename = local_folder + file_blob.name.split('/')[-1]
            file_blob.download_to_filename(local_filename)
            print(f"{file_blob.name}({index}) successfully downloaded")
        if(write_to_file != None):
            original_stdout = sys.stdout
            with open(write_to_file, 'w') as f:
                sys.stdout = f
                print(index)
                sys.stdout = original_stdout
        index += 1
    
    return None
        