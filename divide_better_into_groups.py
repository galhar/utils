import pandas as pd


def read_xl(sheets_name, relevant_sheet_name):
    return pd.read_excel(sheets_name, sheet_name=None)[relevant_sheet_name]


def create_groups(df, n_in_group=3):
    groups = df.iloc[0:0]
    group = df.iloc[0:0]

    number_col_name = 'num'
    name_col_name = 'name'

    while len(df) > 0:
        current_people = df[df[number_col_name] == df[number_col_name].max()]
        left_to_sample = n_in_group - len(group)

        if len(current_people) >= left_to_sample:
            addon = current_people.sample(n=left_to_sample, random_state=1)
            to_concat = [group, addon]
            group = pd.concat(to_concat)
        else:
            group = pd.concat([group,current_people])

        idx_to_remove = []
        for val in group:
            idx_to_remove += df[df['name'] == val].index.to_list()

        if len(group) == n_in_group or len(group) == len(df):
            groups.append(group)
            group = []

        df = df.drop(idx_to_remove)

    return groups


if __name__ == '__main__':
    sheets_name = 'linux workshop level assement.xlsx'
    relevant_sheet_name = 'sheet1'

    df = read_xl(sheets_name, relevant_sheet_name)
    print(create_groups(df))
