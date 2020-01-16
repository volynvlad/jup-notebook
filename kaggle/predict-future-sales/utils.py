def eda(data):
    print("----------Top-5- Record----------")
    print(data.head(5))
    print("-----------Information-----------")
    print(data.info())
    print("-----------Data Types-----------")
    print(data.dtypes)
    print("----------Missing value-----------")
    print(data.isnull().sum())
    print("----------Null value-----------")
    print(data.isna().sum())
    print("----------Shape of Data----------")
    print(data.shape)


def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include=['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);


def drop_duplicate(data, subset):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(subset, keep='first',
                         inplace=True)  # subset is list where you have to put all column for duplicate check
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before - after)


def unresanable_data(data):
    print("Min Value\n{}".format(data.min()))
    print("Max Value\n{}".format(data.max()))
    print("Average Value\n{}".format(data.mean()))
    print("Center Point of Data\n{}".format(data.median()))
