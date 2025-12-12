import time
import json
import os
import numpy as np
from collections import Counter
from scipy import stats


def convert_numpy_types(obj):
    """Recursively convert NumPy data types to native Python types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def compute_signature_similarity(sim_sig, ref_sig):
    """Compute cosine similarity between two signature vectors"""
    sim_norm = sim_sig / (np.linalg.norm(sim_sig) + 1e-8)
    ref_norm = ref_sig / (np.linalg.norm(ref_sig) + 1e-8)
    return np.dot(sim_norm, ref_norm)


def validate_voting_winner_reliability(residuals, sig_matrix, winner_celltype, 
                                       n_subset_trials=10, subset_ratio=0.7,
                                       n_gene_trials=20, gene_ratios=[0.5, 0.7, 0.8, 0.9],
                                       extreme_percentile=95, significance_level=0.05):
    """
    Comprehensively validate the reliability of the voting winner, checking if it's caused by coincidence or extreme values.

    Parameters:
    - residuals: Residual matrix (n_samples, n_genes).
    - sig_matrix: Candidate signature matrix (n_candidates, n_genes).
    - winner_celltype: Name of the winning cell type from voting.
    - n_subset_trials: Number of trials for the subset robustness test.
    - subset_ratio: Proportion of samples used in each subset.
    - n_gene_trials: Number of trials for the gene sensitivity test.
    - gene_ratios: List of gene proportions to test.
    - extreme_percentile: Percentile threshold for extreme value genes.
    - significance_level: Statistical significance level.

    Returns:
    - validation_results: A dictionary containing all validation results.
    """
    
    import numpy as np
    from collections import Counter
    from scipy import stats
    
    print("="*80)
    print(f"Reliability Validation Report for Voting Winner '{winner_celltype}'")
    print("="*80)
    
    validation_results = {
        'winner_celltype': winner_celltype,
        'validations_passed': 0,
        'total_validations': 0,
        'detailed_results': {}
    }
    
    # ========================================================================
    # Test 1: Extreme Value Impact Test
    # ========================================================================
    print("\n1. Extreme Value Impact Test")
    print("-" * 50)
    
    # Identify extremely expressed genes
    gene_max_values = np.max(residuals, axis=0)
    extreme_threshold = np.percentile(gene_max_values, extreme_percentile)
    extreme_genes = gene_max_values > extreme_threshold
    normal_genes = ~extreme_genes
    
    print(f"Detected {extreme_genes.sum()} extreme value genes (>{extreme_percentile}th percentile)")
    
    # Calculate voting results including/excluding extreme genes respectively
    def vote_with_gene_mask(gene_mask, mask_name):
        vote_counts = Counter()
        for residual in residuals:
            similarities = {}
            for candidate in sig_matrix.index:
                ref_sig = sig_matrix.loc[candidate].values
                sim = compute_signature_similarity(residual[gene_mask], ref_sig[gene_mask])
                similarities[candidate] = sim
            
            winner = max(similarities.items(), key=lambda x: x[1])[0]
            vote_counts[winner] += 1
        
        return vote_counts.most_common()
    
    # Vote with all genes
    full_ranking = vote_with_gene_mask(np.ones(len(normal_genes), dtype=bool), "全基因")
    # Vote excluding extreme genes
    normal_ranking = vote_with_gene_mask(normal_genes, "exclude_extreme_genes")
    
    print("Voting Results Comparison:")
    print("Top 5 All Genes vs Top 5 Excluding Extreme Genes")
    for i in range(min(5, len(full_ranking), len(normal_ranking))):
        full_winner, full_votes = full_ranking[i]
        normal_winner, normal_votes = normal_ranking[i]
        print(f"  {i+1}. {full_winner} ({full_votes}) vs {normal_winner} ({normal_votes})")
    
    extreme_validation_passed = (normal_ranking[0][0] == winner_celltype)
    
    validation_results['total_validations'] += 1
    if extreme_validation_passed:
        validation_results['validations_passed'] += 1
        print(f"✓ Extreme Value Test: {winner_celltype} still wins after excluding extreme genes")
    else:
        print(f"✗ Extreme Value Test: {winner_celltype} loses the top position after excluding extreme genes")
    
    validation_results['detailed_results']['extreme_value_test'] = {
        'passed': extreme_validation_passed,
        'full_ranking': full_ranking[:5],
        'normal_ranking': normal_ranking[:5],
        'extreme_genes_count': extreme_genes.sum()
    }
    
    # ========================================================================
    # Test 2: Subset Robustness Test
    # ========================================================================
    print(f"\n2. Subset Robustness Test ({n_subset_trials} trials)")
    print("-" * 50)
    
    n_samples = len(residuals)
    subset_size = int(n_samples * subset_ratio)
    subset_winners = []
    
    for trial in range(n_subset_trials):
        # Randomly select a subset of samples
        subset_indices = np.random.choice(n_samples, subset_size, replace=False)
        subset_residuals = residuals[subset_indices]
        
        # Vote on the subset
        vote_counts = Counter()
        for residual in subset_residuals:
            similarities = {}
            for candidate in sig_matrix.index:
                ref_sig = sig_matrix.loc[candidate].values
                sim = compute_signature_similarity(residual, ref_sig)
                similarities[candidate] = sim
            
            winner = max(similarities.items(), key=lambda x: x[1])[0]
            vote_counts[winner] += 1
        
        subset_winner = vote_counts.most_common(1)[0][0]
        subset_winners.append(subset_winner)
    
    # Calculate consistency
    winner_consistency = Counter(subset_winners)
    consistency_ratio = winner_consistency.get(winner_celltype, 0) / n_subset_trials
    
    print(f"{winner_celltype} won in subset tests: {winner_consistency.get(winner_celltype, 0)}/{n_subset_trials} times")
    print(f"Consistency Ratio: {consistency_ratio*100:.1f}%")
    
    # Show other winners
    if len(winner_consistency) > 1:
        print("Other winners:")
        for celltype, count in winner_consistency.most_common()[1:]:
            print(f"  {celltype}: {count}times ({count/n_subset_trials*100:.1f}%)")
    
    subset_validation_passed = (consistency_ratio >= 0.7)  #70% consistency threshold
    
    validation_results['total_validations'] += 1
    if subset_validation_passed:
        validation_results['validations_passed'] += 1
        print(f"✓ Subset Robustness Test: Passed (Consistency ≥ 70%)")
    else:
        print(f"✗ Subset Robustness Test: Failed (Consistency < 70%)")
    
    validation_results['detailed_results']['subset_robustness_test'] = {
        'passed': subset_validation_passed,
        'consistency_ratio': consistency_ratio,
        'winner_counts': dict(winner_consistency),
        'subset_winners': subset_winners
    }
    
    # ========================================================================
    # Test 3: Gene Subset Sensitivity Analysis
    # ========================================================================
    print(f"\n3. Gene Subset Sensitivity Analysis")
    print("-" * 50)
    
    n_genes = residuals.shape[1]
    gene_sensitivity_results = {}
    
    for ratio in gene_ratios:
        n_selected = int(n_genes * ratio)
        trial_winners = []
        
        for trial in range(n_gene_trials):
            # Randomly select a subset of genes
            selected_genes = np.random.choice(n_genes, n_selected, replace=False)
            
            # Vote using the selected genes
            vote_counts = Counter()
            for residual in residuals:
                similarities = {}
                for candidate in sig_matrix.index:
                    ref_sig = sig_matrix.loc[candidate].values
                    
                    residual_subset = residual[selected_genes]
                    ref_subset = ref_sig[selected_genes]
                    
                    sim = compute_signature_similarity(residual_subset, ref_subset)
                    similarities[candidate] = sim
                
                winner = max(similarities.items(), key=lambda x: x[1])[0]
                vote_counts[winner] += 1
            
            overall_winner = vote_counts.most_common(1)[0][0]
            trial_winners.append(overall_winner)
        
        # Calculate consistency for this gene ratio
        winner_counts = Counter(trial_winners)
        consistency = winner_counts.get(winner_celltype, 0) / n_gene_trials
        
        gene_sensitivity_results[ratio] = {
            'consistency': consistency,
            'winner_counts': dict(winner_counts)
        }
        
        print(f"Using {ratio*100:4.0f}% genes: {winner_celltype} won {winner_counts.get(winner_celltype, 0):2d}/{n_gene_trials} times "
            f"({consistency*100:5.1f}%)")
    
    # Evaluate gene sensitivity
    avg_gene_consistency = np.mean([r['consistency'] for r in gene_sensitivity_results.values()])
    min_gene_consistency = min([r['consistency'] for r in gene_sensitivity_results.values()])
    
    gene_sensitivity_passed = (avg_gene_consistency >= 0.6 and min_gene_consistency >= 0.4)
    
    validation_results['total_validations'] += 1
    if gene_sensitivity_passed:
        validation_results['validations_passed'] += 1
        print(f"✓ Gene Sensitivity Test: Passed (Average Consistency: {avg_gene_consistency:.1%})")
    else:
        print(f"✗ Gene Sensitivity Test: Failed (Average Consistency: {avg_gene_consistency:.1%})")
    
    validation_results['detailed_results']['gene_sensitivity_test'] = {
        'passed': gene_sensitivity_passed,
        'avg_consistency': avg_gene_consistency,
        'min_consistency': min_gene_consistency,
        'results_by_ratio': gene_sensitivity_results
    }
    
    # ========================================================================
    # 验证4: 统计显著性检验
    # ========================================================================
    print(f"\n4. Statistical Significance Test")
    print("-" * 50)
    
    # Calculate similarity distribution for each candidate type
    similarities_by_type = {}
    for candidate in sig_matrix.index:
        ref_sig = sig_matrix.loc[candidate].values
        similarities = []
        for residual in residuals:
            sim = compute_signature_similarity(residual, ref_sig)
            similarities.append(sim)
        similarities_by_type[candidate] = np.array(similarities)
    
    # T-test between the winner and other candidates
    winner_similarities = similarities_by_type[winner_celltype]
    significant_comparisons = 0
    total_comparisons = 0
    
    print(f"T-test of {winner_celltype} vs other candidates:")
    
    for candidate, similarities in similarities_by_type.items():
        if candidate != winner_celltype:
            t_stat, p_value = stats.ttest_ind(winner_similarities, similarities)
            mean_diff = np.mean(winner_similarities) - np.mean(similarities)
            
            total_comparisons += 1
            if p_value < significance_level and mean_diff > 0:
                significant_comparisons += 1
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  vs {candidate:12s}: t={t_stat:7.3f}, p={p_value:.2e}, "
                  f"diff={mean_diff:7.4f} {significance}")
    
    # ANOVA (Analysis of Variance)
    all_similarities = list(similarities_by_type.values())
    f_stat, p_value_anova = stats.f_oneway(*all_similarities)
    
    print(f"\nOverall ANOVA Test: F={f_stat:.3f}, p={p_value_anova:.2e}")
    
    # Evaluate statistical significance
    significance_ratio = significant_comparisons / total_comparisons if total_comparisons > 0 else 0
    statistical_validation_passed = (significance_ratio >= 0.5 and p_value_anova < significance_level)
    
    validation_results['total_validations'] += 1
    if statistical_validation_passed:
        validation_results['validations_passed'] += 1
        print(f"✓ Statistical Significance Test: Passed ({significant_comparisons}/{total_comparisons} significant comparisons)")
    else:
        print(f"✗ Statistical Significance Test: Failed ({significant_comparisons}/{total_comparisons} significant comparisons)")
    
    validation_results['detailed_results']['statistical_significance_test'] = {
        'passed': statistical_validation_passed,
        'significant_comparisons': significant_comparisons,
        'total_comparisons': total_comparisons,
        'significance_ratio': significance_ratio,
        'anova_p_value': p_value_anova,
        'similarities_by_type': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                                 for k, v in similarities_by_type.items()}
    }
    
    # ========================================================================
    # Overall Assessment
    # ========================================================================
    print("\n" + "="*80)
    print("Overall Validation Results")
    print("="*80)
    
    validation_score = validation_results['validations_passed'] / validation_results['total_validations']
    
    print(f"Validation Score: {validation_results['validations_passed']}/{validation_results['total_validations']} ({validation_score:.1%})")  

    # Generate conclusion
    if validation_score >= 0.8:
        conclusion = f"The discovery of '{winner_celltype}' is considered highly reliable"
        reliability = "HIGH"
    elif validation_score >= 0.6:
        conclusion = f"The discovery of '{winner_celltype}' has moderate reliability, cautious interpretation is advised"
        reliability = "MODERATE"
    elif validation_score >= 0.4:
        conclusion = f"The discovery of '{winner_celltype}' has low reliability and requires further validation"
        reliability = "LOW"
    else:
        conclusion = f"The discovery of '{winner_celltype}' is likely unreliable, re-examination is strongly recommended"
        reliability = "UNRELIABLE"
    
    print(conclusion)
    
    # 提供具体建议
    print(f"\nSpecific Validation Status:")
    for i, (test_name, result) in enumerate([
        ("Extreme Value Impact", validation_results['detailed_results']['extreme_value_test']['passed']),
        ("Subset Robustness", validation_results['detailed_results']['subset_robustness_test']['passed']),
        ("Gene Sensitivity", validation_results['detailed_results']['gene_sensitivity_test']['passed']),
        ("Statistical Significance", validation_results['detailed_results']['statistical_significance_test']['passed'])
    ], 1):
        status = "✓ Passed" if result else "✗ Failed"
        print(f"  {i}. {test_name}: {status}")
    
    validation_results['validation_score'] = validation_score
    validation_results['reliability'] = reliability
    validation_results['conclusion'] = conclusion
    
    print("="*80)
    
    return validation_results



# ============================================================================
# 实验可靠性分析模块 - 集成到现有distillation流程
# ============================================================================

def collect_round_data_for_analysis(round_num, voting_results, similarity_rankings, 
                                    residual_intensity, discovered_celltype=None):
    """
    收集每轮的关键数据用于后续分析
    这个函数在每轮发现完成后调用
    """
    if not hasattr(collect_round_data_for_analysis, 'round_data_storage'):
        collect_round_data_for_analysis.round_data_storage = []
    
    # 计算投票统计
    if voting_results and len(voting_results) > 0:
        total_votes = sum(voting_results.values())
        winner = max(voting_results.keys(), key=lambda k: voting_results[k])
        winner_votes = voting_results[winner]
    else:
        # 如果没有voting结果，使用默认值
        total_votes = 0
        winner = discovered_celltype if discovered_celltype else "Unknown"
        winner_votes = 0
    
    round_data = {
        'round': round_num,
        'voting_results': dict(voting_results),  # 复制投票结果
        'winner': winner,
        'winner_votes': winner_votes,
        'total_votes': total_votes,
        'winner_percentage': winner_votes / total_votes if total_votes > 0 else 0,
        'similarity_rankings': dict(similarity_rankings),  # 复制相似性排名
        'residual_intensity': residual_intensity,
        'discovered_celltype': discovered_celltype
    }
    
    collect_round_data_for_analysis.round_data_storage.append(round_data)
    return round_data

def initialize_reliability_analysis(target_celltypes, known_celltypes, experiment_name):
    """
    初始化可靠性分析
    在实验开始时调用
    """
    if not hasattr(initialize_reliability_analysis, 'analysis_config'):
        initialize_reliability_analysis.analysis_config = {
            'target_celltypes': target_celltypes,
            'known_celltypes': known_celltypes,
            'experiment_name': experiment_name,
            'start_time': time.time()
        }
    
    # 清空之前的数据
    if hasattr(collect_round_data_for_analysis, 'round_data_storage'):
        collect_round_data_for_analysis.round_data_storage = []
    
    print(f"Reliability analysis initialized for experiment: {experiment_name}")
    print(f"Target celltypes: {target_celltypes}")
    print(f"Known celltypes: {known_celltypes}")

def generate_final_reliability_report(final_discovered_types, all_signatures, output_dir):
    """
    生成最终的可靠性分析报告
    在实验完全结束时调用
    """
    # 获取存储的配置和数据
    if not hasattr(initialize_reliability_analysis, 'analysis_config'):
        print("Warning: Reliability analysis was not initialized")
        return None
    
    if not hasattr(collect_round_data_for_analysis, 'round_data_storage'):
        print("Warning: No round data was collected")
        return None
    
    config = initialize_reliability_analysis.analysis_config
    round_data = collect_round_data_for_analysis.round_data_storage
    
    print("\n" + "="*80)
    print("GENERATING RELIABILITY ANALYSIS REPORT")
    print("="*80)
    
    # 分析静态特征
    static_features = analyze_static_characteristics(
        config['target_celltypes'], 
        config['known_celltypes'], 
        all_signatures
    )
    
    # 分析动态过程
    dynamics = analyze_discovery_process(config['target_celltypes'], round_data)
    
    # 评估最终结果
    final_assessment = evaluate_discovery_outcomes(
        config['target_celltypes'], 
        final_discovered_types
    )
    
    # 提取关键指标
    key_metrics = extract_reliability_metrics(static_features, dynamics, final_assessment)
    
    # 构建完整报告
    report = {
        'experiment_info': {
            'name': config['experiment_name'],
            'target_celltypes': config['target_celltypes'],
            'known_celltypes': config['known_celltypes'],
            'experiment_duration': time.time() - config['start_time']
        },
        'static_features': static_features,
        'dynamics': dynamics,
        'final_assessment': final_assessment,
        'key_metrics': key_metrics
    }
    
    # 保存报告
    save_detailed_report(report, output_dir, config['experiment_name'])
    
    # 打印摘要
    print_analysis_summary(report)
    
    return report

def analyze_static_characteristics(target_celltypes, known_celltypes, all_signatures):
    """
    分析实验前就能确定的静态特征
    """
    print("\n--- Analyzing Static Characteristics ---")
    
    # 计算你关心的二维坐标
    max_known_risk = 0.0
    inter_unknown_sim = 0.0
    
    # 计算最大已知混淆风险
    for target in target_celltypes:
        if target in all_signatures:
            target_signature = all_signatures[target]
            for known in known_celltypes:
                if known in all_signatures:
                    # 计算余弦相似度
                    similarity = compute_cosine_similarity_pair(target_signature, all_signatures[known])
                    max_known_risk = max(max_known_risk, similarity)
    
    # 计算未知类型间相似性
    if len(target_celltypes) >= 2:
        for i, type1 in enumerate(target_celltypes):
            for type2 in target_celltypes[i+1:]:
                if type1 in all_signatures and type2 in all_signatures:
                    similarity = compute_cosine_similarity_pair(all_signatures[type1], all_signatures[type2])
                    inter_unknown_sim = max(inter_unknown_sim, similarity)
    
    # 计算相似性邻域特征
    neighborhood_analysis = {}
    for target in target_celltypes:
        if target in all_signatures:
            similarities = []
            target_sig = all_signatures[target]
            
            for other_type, other_sig in all_signatures.items():
                if other_type != target:
                    sim = compute_cosine_similarity_pair(target_sig, other_sig)
                    similarities.append((other_type, sim))
            
            # 按相似性排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            neighborhood_analysis[target] = {
                'top_3_similar': similarities[:3],
                'high_similarity_count': len([s for s in similarities if s[1] > 0.7]),
                'medium_similarity_count': len([s for s in similarities if s[1] > 0.5])
            }
    
    coordinates = (max_known_risk, inter_unknown_sim)
    
    print(f"Similarity Coordinates: ({max_known_risk:.3f}, {inter_unknown_sim:.3f})")
    print(f"Max Known Confusion Risk: {max_known_risk:.3f}")
    print(f"Inter-Unknown Similarity: {inter_unknown_sim:.3f}")
    
    return {
        'similarity_coordinates': {
            'max_known_confusion_risk': max_known_risk,
            'inter_unknown_similarity': inter_unknown_sim,
            'coordinate_string': f"({max_known_risk:.3f}, {inter_unknown_sim:.3f})"
        },
        'neighborhood_analysis': neighborhood_analysis
    }

def compute_cosine_similarity_pair(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    import numpy as np
    
    # 确保输入是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def analyze_discovery_process(target_celltypes, round_data):
    """
    分析发现过程的动态特征
    """
    print("\n--- Analyzing Discovery Process ---")
    
    voting_evolution = []
    residual_intensities = []
    
    for round_info in round_data:
        # 分析投票质量
        decision_clarity = calculate_decision_clarity(round_info['voting_results'])
        
        # 分析目标类型表现
        target_performance = {}
        for target in target_celltypes:
            if target in round_info['voting_results']:
                votes = round_info['voting_results'][target]
                percentage = votes / round_info['total_votes'] if round_info['total_votes'] > 0 else 0
                # 计算排名
                sorted_types = sorted(round_info['voting_results'].keys(), 
                                      key=lambda k: round_info['voting_results'][k], reverse=True)
                rank = sorted_types.index(target) + 1 if target in sorted_types else 999
            else:
                votes = 0
                percentage = 0.0
                rank = 999
            
            target_performance[target] = {
                'votes': votes,
                'percentage': percentage,
                'rank': rank,
                'is_winner': target == round_info['winner']
            }
        
        round_analysis = {
            'round': round_info['round'],
            'winner': round_info['winner'],
            'winner_percentage': round_info['winner_percentage'],
            'decision_clarity': decision_clarity,
            'target_performance': target_performance,
            'residual_intensity': round_info['residual_intensity'],
            'discovered_celltype': round_info.get('discovered_celltype')
        }
        
        voting_evolution.append(round_analysis)
        
        if round_info['residual_intensity'] is not None:
            residual_intensities.append(round_info['residual_intensity'])
        
        print(f"Round {round_info['round']}: Winner={round_info['winner']} "
              f"({round_info['winner_percentage']:.1%}), Clarity={decision_clarity:.2f}")
    
    # 分析残差演化
    residual_pattern = analyze_residual_pattern(residual_intensities)
    
    # 分析发现顺序
    discovered_targets = [(r['round'], r['discovered_celltype']) 
                          for r in voting_evolution 
                          if r['discovered_celltype'] in target_celltypes]
    
    return {
        'voting_evolution': voting_evolution,
        'residual_evolution': {
            'intensities': residual_intensities,
            'pattern': residual_pattern
        },
        'discovery_sequence': {
            'discovered_targets': discovered_targets,
            'discovery_count': len(discovered_targets)
        }
    }

def calculate_decision_clarity(voting_results):
    """
    计算决策的清晰程度
    """
    if not voting_results:
        return 0.0
    
    sorted_votes = sorted(voting_results.values(), reverse=True)
    total_votes = sum(sorted_votes)
    
    if len(sorted_votes) == 1 or total_votes == 0:
        return 1.0
    
    # 获胜优势
    winner_advantage = (sorted_votes[0] - sorted_votes[1]) / total_votes
    return winner_advantage

def analyze_residual_pattern(intensities):
    """
    分析残差强度的演化模式
    """
    if len(intensities) < 2:
        return 'insufficient_data'
    
    # 计算变化率
    decline_rates = []
    for i in range(1, len(intensities)):
        if intensities[i-1] > 0:
            rate = (intensities[i-1] - intensities[i]) / intensities[i-1]
            decline_rates.append(rate)
    
    if not decline_rates:
        return 'no_change_data'
    
    avg_decline = sum(decline_rates) / len(decline_rates)
    
    if avg_decline > 0.15:
        return 'healthy_decline'
    elif avg_decline > 0.05:
        return 'moderate_decline'
    elif avg_decline > -0.05:
        return 'stagnant'
    else:
        return 'problematic_pattern'

def evaluate_discovery_outcomes(target_celltypes, discovered_types):
    """
    评估发现结果的质量
    """
    target_set = set(target_celltypes)
    discovered_set = set(discovered_types)
    
    correctly_discovered = target_set & discovered_set
    missed_discoveries = target_set - discovered_set
    false_discoveries = discovered_set - target_set
    
    success_rate = len(correctly_discovered) / len(target_set) if target_set else 0
    perfect_discovery = len(missed_discoveries) == 0 and len(false_discoveries) == 0
    
    return {
        'correctly_discovered': list(correctly_discovered),
        'missed_discoveries': list(missed_discoveries),
        'false_discoveries': list(false_discoveries),
        'success_rate': success_rate,
        'perfect_discovery': perfect_discovery
    }

def extract_reliability_metrics(static_features, dynamics, final_assessment):
    """
    提取关键的可靠性指标
    """
    # 计算平均决策清晰度
    clarity_scores = [r['decision_clarity'] for r in dynamics['voting_evolution']]
    avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
    
    return {
        'coordinates': static_features['similarity_coordinates']['coordinate_string'],
        'max_known_risk': static_features['similarity_coordinates']['max_known_confusion_risk'],
        'inter_unknown_sim': static_features['similarity_coordinates']['inter_unknown_similarity'],
        'avg_decision_clarity': avg_clarity,
        'residual_pattern': dynamics['residual_evolution']['pattern'],
        'discovery_success_rate': final_assessment['success_rate'],
        'perfect_discovery': final_assessment['perfect_discovery'],
        'false_discovery_count': len(final_assessment['false_discoveries'])
    }

def save_detailed_report(report, output_dir, experiment_name):
    """
    保存详细的分析报告
    """
    import json
    import os
    
    # 保存JSON格式的完整报告
    json_path = os.path.join(output_dir, f"{experiment_name}_reliability_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 保存易读的摘要报告
    summary_path = os.path.join(output_dir, f"{experiment_name}_reliability_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT RELIABILITY ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Experiment: {report['experiment_info']['name']}\n")
        f.write(f"Target Celltypes: {report['experiment_info']['target_celltypes']}\n")
        f.write(f"Known Celltypes: {report['experiment_info']['known_celltypes']}\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-" * 20 + "\n")
        metrics = report['key_metrics']
        f.write(f"Similarity Coordinates: {metrics['coordinates']}\n")
        f.write(f"Discovery Success Rate: {metrics['discovery_success_rate']:.2%}\n")
        f.write(f"Perfect Discovery: {'YES' if metrics['perfect_discovery'] else 'NO'}\n")
        f.write(f"Average Decision Clarity: {metrics['avg_decision_clarity']:.3f}\n")
        f.write(f"Residual Evolution Pattern: {metrics['residual_pattern']}\n")
        f.write(f"False Discovery Count: {metrics['false_discovery_count']}\n\n")
        
        f.write("DISCOVERY RESULTS:\n")
        f.write("-" * 20 + "\n")
        final = report['final_assessment']
        f.write(f"Correctly Discovered: {final['correctly_discovered']}\n")
        f.write(f"Missed Discoveries: {final['missed_discoveries']}\n")
        f.write(f"False Discoveries: {final['false_discoveries']}\n")
    
    print(f"Reliability analysis saved to: {json_path}")
    print(f"Summary saved to: {summary_path}")

def print_analysis_summary(report):
    """
    打印分析摘要
    """
    print("\n" + "="*80)
    print("RELIABILITY ANALYSIS SUMMARY")
    print("="*80)
    
    metrics = report['key_metrics']
    final = report['final_assessment']
    
    print(f"Similarity Coordinates: {metrics['coordinates']}")
    print(f"Discovery Success Rate: {metrics['discovery_success_rate']:.2%}")
    print(f"Perfect Discovery: {'YES' if metrics['perfect_discovery'] else 'NO'}")
    print(f"Average Decision Clarity: {metrics['avg_decision_clarity']:.3f}")
    print(f"Residual Evolution: {metrics['residual_pattern']}")
    print(f"False Discoveries: {metrics['false_discovery_count']}")
    
    print(f"\nDetailed Results:")
    print(f"  Correctly Discovered: {final['correctly_discovered']}")
    print(f"  Missed Discoveries: {final['missed_discoveries']}")
    print(f"  False Discoveries: {final['false_discoveries']}")
    
    print("="*80)