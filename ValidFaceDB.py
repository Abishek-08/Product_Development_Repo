from pymongo import MongoClient

def findEmployee(empName):
    client = MongoClient('mongodb://localhost:27017/')
    database = client['face_db']
    collection = database['data']
    
    employeeData = collection.find_one({'empName':empName})
    
    client.close()
    
    return employeeData