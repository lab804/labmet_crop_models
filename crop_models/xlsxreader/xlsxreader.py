#!/usr/bin/env python
# -*- coding: utf-8 -*-

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from itertools import islice

import os
import xlrd


class ExcelDataReader(object):
    """
    Objeto para leitura e parseamento de arquivos do excel
    """
    __pasta = os.path.dirname(os.path.abspath(__file__)).split("crop_models")[0]


    def __init__(self, arquivo_excel, l1, usecols=()):
        print(self.__pasta)
        self.arquivo = arquivo_excel
        self.usecols = usecols
        self.l1 = l1

    @property
    def arquivo(self):
        return self.__arquivo

    @arquivo.setter
    def arquivo(self, arquivo):
        arq = os.path.join(self.__pasta, "crop_models", arquivo)
        self.__arquivo = arq

    def dados(self):
        """
        metodo que retorna os dados para serem utilizados de uma
        planilha de excel
        """

        try:
            sheet = load_workbook(self.arquivo, read_only=True)
            act_sheet = sheet.active
            linhas = act_sheet.rows
            if self.l1 != 0:
                linhas = islice(linhas, self.l1, None)
            dados = []
            for linha in linhas:
                if isinstance(self.usecols, tuple):
                    conteudo = [linha[valor].value for valor in self.usecols]
                else:
                    conteudo = [linha[self.usecols].value]

                if conteudo[0] is not None:
                    dados.append(conteudo)

        except InvalidFileException:
            book = xlrd.open_workbook(self.arquivo)
            sheet = book.sheet_by_index(0)
            dados = []
            for linha in range(self.l1, sheet.nrows, 1):
                conteudo = [sheet.row(linha)[valor].value if isinstance(sheet.row(linha)[valor].value, float)
                            else 0.0 for valor in self.usecols]
                dados.append(conteudo)

        return dados





