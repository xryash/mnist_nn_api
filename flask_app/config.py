class Config(object):
    DEBUG = False
    TESTING = False
    WORKING_FOLDER = ''
    HOST = '0.0.0.0'

class ProductionConfig(Config):
    WORKING_FOLDER = '/home/xryash/local_repository/upload_folder/'


class DevelopmentConfig(Config):
    DEBUG = True
