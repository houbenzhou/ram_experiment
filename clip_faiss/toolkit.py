import sys
def view_bar(num, total):
    """
    进度条
    :param num: 当前进度
    :param total: 任务总量
    :return:
    """
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d' % (">" * rate_num, "-" * (100 - rate_num), rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush()