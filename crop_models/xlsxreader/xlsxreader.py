#!/usr/bin/env python
# -*- coding: utf-8 -*-

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from itertools import islice

import xlrd


class ExcelDataReader(object):
    """
    Objeto para leitura e parseamento de arquivos do excel
    """

    def __init__(self, arquivo_excel, l1, usecols=()):
        self.arquivo = arquivo_excel
        self.usecols = usecols
        self.l1 = l1

    def data(self):
        """
        metodo que retorna os dados para serem utilizados de uma
        planilha de excel
        """

        try:
            sheet = load_workbook(self.arquivo, read_only=True)
            act_sheet = sheet.active
            lines = act_sheet.rows
            if self.l1 != 0:
                lines = islice(lines, self.l1, None)
            data = []
            for line in lines:
                if isinstance(self.usecols, tuple):
                    content = [line[value].value for value in self.usecols]
                else:
                    content = [line[self.usecols].value]

                if content[0] is not None:
                    data.append(content)

        except InvalidFileException:
            book = xlrd.open_workbook(self.arquivo)
            sheet = book.sheet_by_index(0)
            data = []
            for line in range(self.l1, sheet.nrows, 1):
                conteudo = [sheet.row(line)[value].value if isinstance(sheet.row(line)[value].value, float)
                            else 0.0 for value in self.usecols]
                data.append(conteudo)

        return data
