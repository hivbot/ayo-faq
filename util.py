import pandas as pd

SHEET_URL_X = "https://docs.google.com/spreadsheets/d/"
SHEET_URL_Y = "/edit#gid="
SHEET_URL_Y_EXPORT = "/export?gid="


def get_id(sheet_url: str) -> str:
    x = sheet_url.find(SHEET_URL_X)
    y = sheet_url.find(SHEET_URL_Y)
    return sheet_url[x + len(SHEET_URL_X) : y] + "-" + sheet_url[y + len(SHEET_URL_Y) :]


def xlsx_url(get_id: str) -> str:
    y = get_id.rfind("-")
    return SHEET_URL_X + get_id[0:y] + SHEET_URL_Y_EXPORT + get_id[y + 1 :]


def read_df(xlsx_url: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_url, header=0, keep_default_na=False)


def split_page_breaks(df, column_name):
    split_values = df[column_name].str.split("\n")

    new_df = pd.DataFrame({column_name: split_values.explode()})
    new_df.reset_index(drop=True, inplace=True)

    column_order = df.columns

    new_df = new_df.reindex(column_order, axis=1)

    other_columns = column_order.drop(column_name)
    for column in other_columns:
        new_df[column] = (
            df[column].repeat(split_values.str.len()).reset_index(drop=True)
        )

    return new_df


def transform_documents_to_dataframe(documents):
    metadata_keys = set()
    for doc, _ in documents:
        metadata_keys.update(doc.metadata.keys())

    metadata_values = {key: [] for key in metadata_keys}
    for doc, _ in documents:
        for key, value in doc.metadata.items():
            metadata_values[key].append(value)

    metadata_values["Score"] = [score for _, score in documents]

    df = pd.DataFrame(metadata_values)

    return df


def remove_duplicates_by_column(df, column):
    df.drop_duplicates(subset=column, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def dataframe_to_dict(df):
    df_records = df.to_dict(orient='records')

    return df_records