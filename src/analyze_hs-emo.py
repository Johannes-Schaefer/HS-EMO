from src.data import process_data


FOLDER = '../data/'


def print_tex_table_stats(stats_dict, data):
    # print counts and %-values
    header = ''
    vals = data
    for k, v in sorted(stats_dict.items(), key=lambda p: p[1], reverse=True):
        header += '&' + k
        vp = str(round(100 * v / sum(stats_dict.values())))
        vals += '& \\bd{' + str(v) + '}{' + vp + '}'
    print(header + '\\\\')
    print(vals + '\\\\')


def print_tex_table_stats_nested(stats_dict):
    # print counts and %-values
    header = ''
    vals = ''
    for hof_label, emo_dict in sorted(stats_dict.items(), key=lambda p: sum(p[1].values()), reverse=True):
        header += '&' + hof_label
        row_vals = ''
        for emo_key in ('a', 'd', 't', 'h', 'f', 's', '?', ''):
            v = emo_dict[emo_key]
            vp = str(round(100 * v / sum(emo_dict.values())))
            row_vals += '& \\bd{' + str(v) + '}{' + vp + '}'
        vals += row_vals + '\\\\\n'
    print(header + '\\\\')
    print(vals)


if __name__ == '__main__':
    annot_file = FOLDER + 'HS-Emo - Emotionsannotation_1k.csv'
    hs_data_file = FOLDER + 'HS-Emo - en_Hasoc2021_train.csv'
    emo_data_file = FOLDER + 'HS-Emo - tec_dataset.csv'

    hs_data, hs_labels_cg, hs_labels_fg = process_data.read_hasoc_data(hs_data_file)
    hs_emo_data, emo_n_labels, hs_emo_labels_cg, hs_emo_labels_fg, emo_hs_labels_cg, emo_hs_labels_fg\
        = process_data.read_hs_emo_data(annot_file, hs_data, number_of_annotated_instances=1000)
    _, emo_labels = process_data.read_tec_data(emo_data_file)

    # hs label distr
    print_tex_table_stats(hs_labels_cg, 'HASOC')
    print_tex_table_stats(hs_emo_labels_cg, 'HS-EMO')
    print()

    print_tex_table_stats(hs_labels_fg, 'HASOC')
    print_tex_table_stats(hs_emo_labels_fg, 'HS-EMO')
    print()

    # emo label distr
    print_tex_table_stats(emo_labels, 'TEC')
    print_tex_table_stats(emo_n_labels, 'HS-EMO')
    print()

    # hs-emo label distr
    print_tex_table_stats_nested(emo_hs_labels_cg)
    print_tex_table_stats_nested(emo_hs_labels_fg)
