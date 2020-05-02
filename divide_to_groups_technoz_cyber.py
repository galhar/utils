import pandas as pd
from create_xl import create_excel_from_list
import copy
from random import shuffle


def read_xl(sheets_name, relevant_sheet_name):
    return pd.read_excel(sheets_name, sheet_name=None)[relevant_sheet_name]


def create_groups(df, n_in_group=3):
    groups = []
    group = []

    number_col_name = 'num'
    name_col_name = 'name'

    while len(df) > 0:
        current_people = df[df[number_col_name] == df[number_col_name].max()]
        left_to_sample = n_in_group - len(group)

        if len(current_people) >= left_to_sample:
            group += current_people.sample(n=left_to_sample, random_state=1)[name_col_name].to_list()
        else:
            group += current_people[name_col_name].to_list()

        idx_to_remove = []
        for val in group:
            idx_to_remove += df[df['name'] == val].index.to_list()

        if len(group) == n_in_group or len(group) == len(df):
            groups.append(group)
            group = []

        df = df.drop(idx_to_remove)

    return groups


def create_leagues(groups, league_num=3):
    g_copy = copy.deepcopy(groups)

    leagues = []
    n_in_league = int(len(groups) / league_num)
    for i in range(league_num):

        # if it's the last league add all that's left
        if i == league_num - 1:
            shuffle(g_copy)
            leagues.append(g_copy)
            break

        league =g_copy[:n_in_league]
        shuffle(league)

        leagues.append(league)
        g_copy = g_copy[n_in_league:]


    return leagues


if __name__ == '__main__':
    sheets_name = 'networks workshop level assesment.xlsx'
    relevant_sheet_name = 'תגובות לטופס 1'

    df = read_xl(sheets_name, relevant_sheet_name)
    groups = create_groups(df)
    leagues = create_leagues(groups)

    create_excel_from_list(leagues, 'קבוצות סדנת רשתות')
