import json
import sys

import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np
import yaml

from attacks.strategies import get_strategy


def _dump_success_rate_plot(results_df, strategy_index_name_map, out_path):
    # plot of exploration_budget vs attack success rate
    success_rate_df = results_df.groupby(['strategy_index', 'used_budget'])[['is_correct']].mean().reset_index()
    success_rate_df['success_rate'] = 1 - success_rate_df['is_correct']
    success_rate_df['strategy_name'] = success_rate_df['strategy_index'].map(strategy_index_name_map)
    print('Maximal success rate:')
    max_sr = success_rate_df.iloc[success_rate_df['success_rate'].argmax()][['success_rate', 'strategy_name', 'used_budget']].to_dict()
    max_sr['used_budget'] = int(max_sr['used_budget'])
    print(max_sr)
    fig = px.line(success_rate_df, x="used_budget", y="success_rate", color='strategy_name')
    fig.to_plotly_json()
    if out_path is not None:
        fig.write_html(out_path + 'html')
        fig.write_json(out_path + 'json')
        success_rate_df.to_csv(out_path + 'csv')
    return fig


def _dump_confidence_drop_plot(results_df, orig_results_df, strategy_index_name_map, out_path):
    # plot of exploration_budget vs drop in relative model confidence
    joined_df = results_df.set_index('sample_index').join(orig_results_df.set_index('sample_index'))
    joined_df['relative_drop'] = (joined_df['orig_loss'] - joined_df['attack_loss']) / joined_df['orig_loss']
    grouped_joined_df = joined_df.groupby(['strategy_index', 'used_budget'])[['relative_drop']].mean().reset_index()
    grouped_joined_df['strategy_name'] = grouped_joined_df['strategy_index'].map(strategy_index_name_map)
    fig = px.line(grouped_joined_df, x="used_budget", y="relative_drop", color='strategy_name')
    if out_path is not None:
        fig.write_html(out_path + 'html')
        fig.write_json(out_path + 'json')
    return fig


def _dump_winners_plot(results_df, strategy_index_name_map, out_path, analyze_winners_locs):
    results = results_df[results_df.used_budget.isin(analyze_winners_locs)]
    if len(results) == 0:
        print('no winners to plots at the analyzed locs')
        return None
    results = results.loc[results.groupby(['sample_index', 'used_budget'])['attack_loss'].idxmin()]
    results['strategy_name'] = results['strategy_index'].map(strategy_index_name_map)
    results = results.groupby(['used_budget', 'strategy_name'])['is_correct'].count().reset_index()
    fig = px.bar(results.sort_values('is_correct', ascending=False), x="used_budget", y="is_correct", color="strategy_name", title="Winners analysis")
    if out_path is not None:
        fig.write_html(out_path + 'html')
        fig.write_json(out_path + 'json')
    return fig


def __space_size_to_bin(ssize: int) -> str:
    p = np.ceil(np.log10(ssize))
    if p == 0:
        return '1'
    low = f'1e{p-1:02.0f}' if p > 1 else '1'
    return f'{low}<  <=1e{p:02.0f}'


def __positions_size_to_bin(ssize: int) -> str:
    if ssize <= 10:
        return f'{ssize:03.0f}'
    lower_bound = 10 * int(ssize / 10)
    upper_bound = lower_bound + 10
    return f'{lower_bound:03.0f}<=  <{upper_bound:03.0f}'


def _dump_attack_space_results_plot(results_df, attack_space_size, strategy_index_name_map, out_path, binner_func=__space_size_to_bin):
    results = results_df[results_df.used_budget == results_df.used_budget.max()].copy()
    results['attack_space'] = results.apply(lambda row: binner_func(attack_space_size.loc[row.sample_index, row.strategy_index]), axis=1)
    results['strategy_name'] = results['strategy_index'].map(strategy_index_name_map)
    results = results.groupby(['attack_space', 'strategy_name']).agg(fail=('is_correct', 'sum'), total=('is_correct', 'count')).reset_index()
    results['success'] = results.total - results.fail
    x = sorted(results.attack_space.unique())

    data = []

    for i, (s_name, df) in enumerate(results.groupby('strategy_name')):
        correct = [0] * len(x)
        fail = [0] * len(x)
        for j, x_ in enumerate(x):
            v = df[df.attack_space == x_]
            if len(v) > 0:
                v = v.iloc[0]
                correct[j] = v.success
                fail[j] = v.fail
        data.append(go.Bar(name=f'{s_name}-Success', x=x, y=correct, offsetgroup=i, legendgroup=f"group{i}", visible='legendonly'))
        data.append(go.Bar(name=f'{s_name}-Fail', x=x, y=fail, offsetgroup=i, base=correct, legendgroup=f"group{i}", visible='legendonly'))

    # Change the bar mode
    # https://github.com/plotly/plotly.py/issues/812
    fig = go.Figure(data=data, layout=go.Layout(title="Analysis of final success rate against attack space size",
                                                xaxis={'tickangle': 45, 'type':'category'}))
    if out_path is not None:
        fig.write_html(out_path + 'html')
        fig.write_json(out_path + 'json')
    return fig


def dump_all_plots(results_df, orig_results_df, attack_space_size, attack_space_positions, strategy_index_name_map, out_dir, exploit_steps,
                   analyze_winners_locs):
    print('Start plotting')
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    results = results_df.copy()
    results['used_budget'] += (exploit_steps - results['used_budget'] % exploit_steps) % exploit_steps
    results['new_index'] = results.apply(lambda row: f'{row.strategy_index:04.0f}-{row.sample_index:04.0f}-{row.used_budget:04.0f}', axis=1)
    results = results.drop_duplicates(subset=['new_index'], keep='last')
    all_samples = sorted(results.sample_index.unique())
    all_budgets = sorted(results.used_budget.unique())
    missing_budgets = set(analyze_winners_locs).difference(analyze_winners_locs)
    assert len(missing_budgets) == 0, f'Budgets {missing_budgets} were requested for winners analyses but they do not exist in the df'
    all_indices = sorted([f'{strategy_index:04.0f}-{sample_index:04.0f}-{used_budget:04.0f}' for strategy_index in results.strategy_index.unique()
                   for sample_index in all_samples for used_budget in all_budgets])
    # results = results.set_index('new_index').sort_index().reindex(all_indices, method='ffill')

    # when explore size > 1 we will have samples that used budget larger than the minimal budget, and thus the inserted rows above them
    # (same strategy and sample num, lower budget) will have NaN but due to the ffill they will get results from the previous line
    # which is obviously wrong. It may mess up with all of the results as there may be inconsistencies in successes (i.e. attack succeeded
    # with budget x but then failed in x+1) and if it happens for the first sample, we will have rows of NaNs.
    cols = results_df.columns.tolist()
    results = results.set_index('new_index').sort_index().reindex(all_indices)
    results['group_idx'] = results.apply(lambda row: f'{row.name[:9]}', axis=1)
    results = results.groupby('group_idx')[cols].ffill()  #.fillna(0)
    results.is_correct.fillna(True, inplace=True)  # We assume the model was correct up until there and that the attacker loss was maximal
    results.attack_loss.fillna(1., inplace=True)  # We assume the model was correct up until there and that the attacker loss was maximal
    results['strategy_index'] = results.apply(lambda row: int(row.name[:4]), axis=1)
    results['sample_index'] = results.apply(lambda row: int(row.name[5:9]), axis=1)
    # now, we have an issue where the used_budget is also forward filled
    results['used_budget'] = results.apply(lambda row: int(row.name[10:]), axis=1)

    # results['used_budget'] = results.index.map(lambda v: int(v.split('-')[-1]))

    plots = {}

    print('dumping success rate plot')
    outpath = os.path.join(out_dir, 'success_rate.') if out_dir is not None else None
    plots['success_rate'] = _dump_success_rate_plot(results, strategy_index_name_map, outpath)
    print('dumping confidence drop plot')
    outpath = os.path.join(out_dir, 'confidence_drop.') if out_dir is not None else None
    plots['confidence_drop'] = _dump_confidence_drop_plot(results, orig_results_df, strategy_index_name_map, outpath)
    print('dumping winners plot')
    outpath = os.path.join(out_dir, 'winners.') if out_dir is not None else None
    plots['winners'] = _dump_winners_plot(results, strategy_index_name_map, outpath, analyze_winners_locs)
    print('dumping attack space plot')
    outpath = os.path.join(out_dir, 'attack_space.') if out_dir is not None else None
    plots['attack_space'] = _dump_attack_space_results_plot(results, attack_space_size, strategy_index_name_map, outpath)
    print('dumping attack positions plot')
    outpath = os.path.join(out_dir, 'attack_positions.') if out_dir is not None else None
    plots['attack_positions'] = _dump_attack_space_results_plot(results, attack_space_positions, strategy_index_name_map, outpath,
                                                                binner_func=__positions_size_to_bin)

    return plots


def load_plot_from_json(fpath):
    with open(fpath) as f:
        fig = json.load(f)
    return go.Figure(data=fig['data'], layout=fig['layout'])


def replot_dir(data_dir, force=True):
    strategies_path = os.path.join(data_dir, 'strategies.yaml')
    with open(strategies_path) as f:
        strategies_confs = yaml.load(f, Loader=yaml.FullLoader)
    try:
        strategy_index_name_map = {i: str(get_strategy(strategy_name, **strategy_values)) for i, strategy in enumerate(strategies_confs)
                                   for strategy_name, strategy_values in strategy.items()}
    except AssertionError:
        strategy_index_name_map = {i: strategy_values.get('name', strategy_name) for i, strategy in enumerate(strategies_confs)
                                   for strategy_name, strategy_values in strategy.items()}

    plots = {
        'success_rate': 'success_rate.json',
        'confidence_drop': 'confidence_drop.json',
        'winners': 'winners.json',
        'attack_space': 'attack_space.json',
        'attack_positions': 'attack_positions.json',
    }
    plots = {k: os.path.join(data_dir, v) for k, v in plots.items()}
    all_plots_exists = all(os.path.exists(v) for v in plots.values())
    if force or not all_plots_exists:
        results = pd.read_csv(os.path.join(data_dir, 'results.csv'))
        orig_results = pd.read_csv(os.path.join(data_dir, 'orig_results.csv'))
        attack_space_size = pd.read_csv(os.path.join(data_dir, 'attack_space.csv'), index_col=0)
        attack_positions_size = pd.read_csv(os.path.join(data_dir, 'attack_positions.csv'), index_col=0)
        with open(os.path.join(data_dir, 'conf.yaml')) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        exploit_steps = args['exploit_every']
        analyze_winners = args['analyze_winners_locs']

        attack_space_size.columns = list(range(len(strategy_index_name_map)))
        attack_positions_size.columns = list(range(len(strategy_index_name_map)))
        plots = dump_all_plots(results, orig_results, attack_space_size, attack_positions_size, strategy_index_name_map, data_dir,
                               exploit_steps, analyze_winners)
    else:
        plots = {k: load_plot_from_json(v) for k, v in plots.items() if v is not None}
    return plots, strategy_index_name_map


if __name__ == '__main__':
    replot_dir(sys.argv[1])
