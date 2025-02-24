from typing import Optional, Union
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from Enum.TestLibraryDataStatusType import TestLibraryDataStatusType
from helper import utility
from Enum.API_Status import APIStatus
from helper.data_status_type_helper import convert_test_library_data_status_type_to_str
from mongoDB.collection_list import db_collection_list
from helper.config_import import mongodb_run
import json


class MongodbController:
    def __init__(self):
        self.db = None
        self.__collection = None
        self.final_query = None
        self.__db_name = "my_db"
        self.__collection_name = {}
        # Replace the placeholder with your Atlas connection string
        self.__mongo_db_url = "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.3"

        self.__set_db_collection()
        self.connect_mongodb()


    def get_db_collection_list(self):
        return self.db.list_collection_names()

    def get_db_document_by_collection(self, collection):
        document_list = []
        collection = self.db.get_collection(collection)
        doc = collection.find({})

        for d in doc:
            d.pop('_id')
            document_list.append(d)

        return document_list


    def __set_db_collection(self):
            self.__collection_name =  {
                db_collection_list.main: db_collection_list.main.value,
                db_collection_list.primary_key : db_collection_list.primary_key.value
            }



    def connect_mongodb(self):
        if mongodb_run:
            # Set the Stable API version when creating a new client
            client =  MongoClient(self.__mongo_db_url,
                                  server_api=ServerApi('1'),
                                  directConnection=True)

            self.db = client[self.__db_name]

    def create_collection(self, collection:db_collection_list):
        if mongodb_run:
            try:
                if collection.value not in self.db.list_collection_names():
                    self.__collection = self.db[collection.value]
                else:
                    self.__collection = self.db.get_collection(collection.value)


            except Exception as e:
                utility.except_error_print(e)


    async def insert_data(self,
                          data:str,
                          search_queries:dict,
                          query_id: Optional[int] = None):
        if mongodb_run:
            db_collection = None
            dd = json.loads(data)

            # adding a custom or auto-generated object id
            self.__attach_query_id_to_data(query=dd, query_id= query_id)
            db_collection = self.__get_collection__(dd)
            search_query = self.__get_query_only_for_id(query_id) if query_id is not None else search_queries

            document_count = self.__get_document_count(db_collection, search_query)
            if document_count == 0:

                try:
                    if isinstance(dd, list):
                        db_collection.insert_many(self.final_query)
                    elif not isinstance(dd, list):
                        db_collection.insert_one(self.final_query)
                    else:
                        utility.except_error_print('no data to insert')

                except Exception as e:
                    utility.except_error_print(e)
            elif document_count == 1:
                # This is only considering replacing one to one
                #await self.update_document_status(TestLibraryDataStatusType.updated)
                update_query =json.dumps(dict(data_status_key=convert_test_library_data_status_type_to_str(
                                    TestLibraryDataStatusType.updated)))

                #await self.delete_data_many()
                self.__update_one(db_collection, search_query, update_query)
            else:
                update_query = json.dumps(dict(data_status_key=convert_test_library_data_status_type_to_str(
                    TestLibraryDataStatusType.updated)))

                # await self.delete_data_many()
                self.__update_many(db_collection, search_query, update_query)


    async def replace_update_one(self,
                                 filter_query:dict,
                                 new_query:dict):
        if mongodb_run:
            return
            # get id
            #current_query_id =filter_query.get('_id') #self.__find_id_by_query(filter_query)
            #id_query = self.__get_query_only_for_id(current_query_id)

            # get a document for the id
            #current_query_document : dict = self.__collection.find_one(id_query)
            #update_query_document = dict(current_query_document)
            #update_query_document.update(new_query)
            #await self.delete_data_many(query_id=int(current_query_id))
            # try:
            #     self.__collection.replace_one(current_query_document,
            #                                   update_query_document)
            #
            # except Exception as e:
            #     utility.except_error_print(e)



    async def update_one(self,
                               search_query : str,
                               update_query:str):
        '''

        :param filter_query:
        :param new_query:
        :return:
        '''
        if mongodb_run:
            dd = json.loads(search_query)
            db_collection = self.__get_collection__(dd)

            if isinstance(dd, list):
                for d in dd:
                    self.__update_one(db_collection, d, update_query)
            else:
                self.__update_one(db_collection, dd, update_query)

            #_update_query = {'$set': json.loads(update_query)}
            #db_collection.update_one(dd, _update_query)


            #await self.delete_data_many(query_id=query_id)
            # if await self.get_document_count(filter_query) == 1:
            #     #self.__collection.update_one()
            #     old_document = self.__collection.find_one(filter_query)
            #     _id = self.__find_id_by_query(filter_query)
            #     await self.delete_data_many(query_id=int(_id))
            #     await self.insert_data(query_id= int(_id),data=filter_query)
            #     #self.__collection.update_one()
            #     # #new_query.update({"_id": _id}) # add an id on a new query
            #     # _new_query = dict(new_query)
            #     # self.__collection.replace_one(old_document, _new_query)
            # else:
            #     print(f"{filter_query} has more data in the database")

    def __update_one(self, collection, find_query:dict, update_query:str):
        _update_query = {'$set': json.loads(update_query)}
        collection.update_one(find_query, _update_query)

    def __update_many(self, collection, find_query: dict, update_query: str):
        _update_query = {'$set': json.loads(update_query)}
        collection.update_many(find_query, _update_query)

    def __get_collection__(self, data:[dict, list]):
        """
        get a collection for current data
        :param data:
        :return:
        """
        keys = data[0].keys() if isinstance(data, list) else data.keys()
        # crate a new collection if not existing
        if 'collection' in keys:
            db_collection_name = data[0].pop('collection') if isinstance(data, list) else data.pop('collection')
            db_collection = self.get_collection_by_name(db_collection_name)
        else:
            db_collection = self.__collection

        return db_collection

    async def find(self,
                   query:list):
        if mongodb_run:
            document_list = []

            doc = self.__collection.find(query)

            for d in doc:
                document_list.append(d)

            return document_list

    async def find_all(self):
        if mongodb_run:
            document_list = []

            for d in self.__collection.find({}):
                document_list.append(d)

            return document_list
        else:
            return None

    async def update_document_status(self, data_stat_type: TestLibraryDataStatusType):
        if mongodb_run:
            if not isinstance(self.final_query, dict):
                utility.except_error_print("final query should be a dictionary type")

            value  = convert_test_library_data_status_type_to_str(data_stat_type)

            self.final_query.update(dict(data_status_key = value))


    async def get_data(self,
                       query: Optional[str] = None,
                       query_id:Optional[int]=None) -> list:

        if mongodb_run:

            dd = json.loads(query)


            document_list=[]

            # find_query = ''
            # for key in list(query.keys()):
            #     find_query = find_query + '.' + key

            if 'collection' in dd.keys():
                db_collection_name = dd.pop('collection')
                db_collection = self.get_collection_by_name(db_collection_name)
            else:
                db_collection = self.__collection


            self.__attach_query_id_to_data(query=dd, query_id=query_id)
            doc = db_collection.find(self.final_query)


            for d in doc:
                document_list.append(d)



            # for document in self.__collection.find(query):
            #     document_list.append(document)
            return document_list


    async def get_value_by_field(self,
                       filter_query: str,
                       field:str) -> Union[str, dict]:
        """
        this returns a value with searched query
        :param filter_query: It would look upto a data with the filter_query
        :param field: It returns a value of the field from results that were looked up.
        :return:
        """

        if mongodb_run:
            _filter_query = json.loads(filter_query)
            if 'collection' in _filter_query.keys():
                db_collection_name = _filter_query.pop('collection')
                db_collection = self.get_collection_by_name(db_collection_name)
            else:
                db_collection = self.__collection

            self.__attach_query_id_to_data(query=_filter_query)

            doc_count = self.__get_document_count(db_collection, self.final_query)
            if doc_count == 1:
                return db_collection.find_one(self.final_query).get(field)
            elif doc_count > 0:
                changed_record_count = len([d for d in db_collection.find({}) if d.get(field) != convert_test_library_data_status_type_to_str(TestLibraryDataStatusType.updated)])
                return convert_test_library_data_status_type_to_str(TestLibraryDataStatusType.modified) \
                            if changed_record_count > 0 \
                            else convert_test_library_data_status_type_to_str(TestLibraryDataStatusType.updated)
            else:
                return convert_test_library_data_status_type_to_str(TestLibraryDataStatusType.Nodata)



    async def delete_data_many(self,
                               query: Optional[dict] = None,
                               query_id:Optional[int] = None):
        if mongodb_run:
            if query is None and query_id is None:
                raise 'no parameter'

            try:
                if query_id is None and query is not None:
                    self.final_query = dict(query)
                elif query_id is not None and query is None:
                    self.final_query = self.__get_query_only_for_id(query_id)
                else:
                    self.__attach_query_id_to_data(query=query, query_id=query_id)

                if self.__collection.count_documents(self.final_query) == 0:
                    print('Collection does not have this data\n' + f'{query}\n')
                else:
                    self.__collection.delete_many(self.final_query)
                    print(f'data is deleted (Id : {query_id})')
            except Exception as e:
                utility.except_error_print(e)


    async def remove_collections(self,
                                 query:dict) -> APIStatus:
        if mongodb_run:
            try:

                if query is None:
                    for collection in self.db.list_collection_names():
                        self.db.get_collection(collection).delete_many({})
                else:
                    collection = query.pop("collection_name")
                    self.get_collection_by_name(collection).delete_many({})

                return APIStatus.SUCCESS
            except Exception as e:
                utility.except_error_print(e)
                return APIStatus.FAIL


    async def get_document_count(self,
                                 search_query:str):
        dd = json.loads(search_query)

        if 'collection' in dd.keys():
            db_collection_name = dd.pop('collection')
            db_collection = self.get_collection_by_name(db_collection_name)
        else:
            db_collection = self.__collection

        return self.__get_document_count(db_collection)





    def __find_id_by_query(self,
                           filter_query:dict) -> str:
        """
        Return an id for the filtered query
        :param filter_query:
        :return:
        """
        old_document = self.__collection.find_one(filter_query)
        # id for the filtered query
        return old_document.get('_id')

    def __get_document_count(self,
                             db_collection = None,
                             data:Optional[dict]= None) -> int:
        if mongodb_run:
            try:
                _data = {} if data is None else data
                if db_collection is not None:
                    count = db_collection.count_documents(_data)
                else:
                    count = self.__collection.count_documents(_data)

                return count
            except Exception as e:
                utility.except_error_print(e)

    def __attach_query_id_to_data(self,
                                  query:Union[list, dict] = None,
                                  query_id: Optional[int] = None):
        if query is not None and not isinstance(query, list) and  not isinstance(query, dict):
            utility.except_error_print('no list, dict or string type')

        #_query = dict(query) if isinstance(query, dict) else list(query)
        if query is not None:
            _query : Union[dict, list]

            try:
                # if the passed query is a list, it won't need to copy it to a new dict
                if not isinstance(query, list):
                    _query = {}
                    _query.update(query)
                else:
                    _query = []
                    [_query.append(d) for d in query]


            except Exception as e:
                utility.except_error_print(e)

            try:
                # if there is a designated query id
                if query_id is not None:
                    _query_id = self.__get_query_only_for_id(query_id)

                    if not isinstance(query, list):
                        _query.update(_query_id)
                        self.final_query = _query
                    else:

                        _list_query = query.copy()
                        self.final_query = _list_query
                else:
                    self.final_query = _query

            except Exception as e:
                utility.except_error_print(e)
        else:
            self.final_query = self.__get_query_only_for_id(query_id)




    def __get_query_only_for_id(self, query_id) \
            -> dict:
        return {'_id': query_id}


    # async def ping_server(client):
    #
    #     # Send a ping to confirm a successful connection
    #     try:
    #         client.admin.command('ping')
    #         print("Pinged your deployment. You successfully connected to MongoDB!")
    #     except Exception as e:
    #         print(e)

    def get_collection_by_name(self,
                       collection_name:str):
        if collection_name not in self.db.list_collection_names():
            try:
                self.db.create_collection(collection_name)
            except Exception as e:
                utility.except_error_print(e)

        return self.db.get_collection(collection_name)












