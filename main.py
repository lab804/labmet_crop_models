from crop_models.xlsxreader.xlsxreader import ExcelDataReader

data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                       usecols=(0, 1, 2,3,4,5,6,7,8, 9))

print(data.data())