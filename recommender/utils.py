import random


def epsilon_greedy_selection(
    candidate_ids, candidate_scores, epsilon=0.1, n_select=100
):
    """
    Chọn n_select ứng viên từ candidate_ids sử dụng chiến lược epsilon-greedy.

    Args:
        candidate_ids (list hoặc np.array): Danh sách các ID ứng viên (200 bất động sản).
        candidate_scores (list hoặc np.array): Điểm số tương ứng với từng ứng viên.
        epsilon (float): Xác suất khám phá (exploration). Mặc định là 0.1.
        n_select (int): Số lượng ứng viên cần chọn (mặc định 100).

    Returns:
        list: Danh sách candidate_ids được chọn.
    """
    remaining_indices = list(range(len(candidate_ids)))
    selected_indices = []

    while len(selected_indices) < n_select and remaining_indices:
        if random.random() < epsilon:
            chosen_index = random.choice(remaining_indices)
        else:
            chosen_index = max(remaining_indices, key=lambda idx: candidate_scores[idx])
        selected_indices.append(chosen_index)
        remaining_indices.remove(chosen_index)
    selected_candidate_ids = [candidate_ids[i] for i in selected_indices]
    return selected_candidate_ids
