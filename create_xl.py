import xlsxwriter


def get_longest_length(groups_list):
    return max([sum([len(name) for name in group]) for group in groups_list])


def create_excel_from_list(leagues, file_name, leagues_names=None):
    file = file_name + '.xlsx'
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(file)
    worksheet = workbook.add_worksheet()

    # create formats
    title_f = workbook.add_format({'bold': True, 'text_wrap': True, 'border': 2, 'fg_color': 'EAFFEA'})
    def_f = workbook.add_format({'text_wrap': True, 'border': 1})
    num_f = workbook.add_format({'text_wrap': True, 'border': 1})
    num_f.set_right(2)

    if len(leagues) == 1:
        leagues_names = [file_name]
    elif leagues_names is None:
        leagues_names = ['ליגה ' + str(i + 1) for i in range(len(leagues))]

    start_col, start_line = 'A', 1
    # first_group_num = 1
    for i, league in enumerate(leagues):
        insert_league(def_f, league, num_f, title_f, worksheet, start_col, start_line,
                      leagues_names[i])

        # group_col_width = get_longest_length(league)
        # # worksheet.set_column(first_col=ord(start_col) - ord('A'), last_col=1, width=group_col_width)
        # worksheet.set_column(first_col=0, last_col=0, width=len(group_num_col_title))

        n_in_group = max([len(g) for g in league])
        # first_group_num += n_in_group
        start_col = chr(ord(start_col) + n_in_group + 2)

    workbook.close()


def insert_league(def_f, groups_list, num_f, title_f, worksheet, begin_col, begin_line, league_title):
    nun_in_group = max([len(g) for g in groups_list])

    # league title
    league_title_range_str = begin_col + str(begin_line) + ':' + chr(ord(begin_col) + nun_in_group) + str(
        begin_line)
    worksheet.merge_range(league_title_range_str, league_title, title_f)

    # write the title for the groups
    group_num_col_title = 'מספר קבוצה'
    worksheet.write(begin_col + str(begin_line + 1), group_num_col_title, title_f)

    # inserts the participants
    group_friends_range_str = chr(ord(begin_col) + 1) + str(begin_line + 1) + ':' + chr(
        ord(begin_col) + nun_in_group) + str(begin_line + 1)

    worksheet.merge_range(group_friends_range_str, "חברי הקבוצה", title_f)
    insert_groups(def_f, groups_list, begin_line + 2, num_f, worksheet, begin_col)


def insert_groups(def_f, groups_list, line_gap, num_f, worksheet, begin_col):
    for i, group in enumerate(groups_list):
        # group num
        worksheet.write(begin_col + str(i + line_gap), int(i + 1), num_f)

        insert_participants(def_f, group, i, line_gap, worksheet, begin_col=chr(ord(begin_col) + 1))


def insert_participants(def_f, group, i, line_gap, worksheet, begin_col='B'):
    """

    :param def_f: format for the names
    :param group: the group
    :param i: the group number or row number
    :param line_gap: the first line
    :param worksheet:
    :param begin_col: the columns from which to insert the group
    :return:
    """
    for j, name in enumerate(group):
        worksheet.write(chr(ord(begin_col) + j) + str(i + line_gap), name, def_f)
