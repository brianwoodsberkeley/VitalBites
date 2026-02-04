const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Helper to get auth headers
const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

// Register a new user
export const register = async (email, password, ailmentIds) => {
  const response = await fetch(`${API_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      password,
      ailment_ids: ailmentIds,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Registration failed');
  }

  return response.json();
};

// Login user
export const login = async (email, password) => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  const response = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }

  const data = await response.json();
  localStorage.setItem('token', data.access_token);
  return data;
};

// Logout user
export const logout = () => {
  localStorage.removeItem('token');
};

// Get current user profile
export const getCurrentUser = async () => {
  const response = await fetch(`${API_URL}/users/me`, {
    headers: {
      ...getAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error('Failed to get user profile');
  }

  return response.json();
};

// Check if user is logged in
export const isLoggedIn = () => {
  return !!localStorage.getItem('token');
};

// Get recipe recommendations
export const getRecommendations = async (count = 10) => {
  const response = await fetch(`${API_URL}/recommendations?count=${count}`, {
    headers: {
      ...getAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error('Failed to get recommendations');
  }

  return response.json();
};

// Submit recipe feedback
export const submitFeedback = async (recipe, cooked = false, skipped = false, rating = null) => {
  const response = await fetch(`${API_URL}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      recipe_id: recipe.id,
      recipe_name: recipe.name,
      recipe_image: recipe.image,
      recipe_data: recipe,
      cooked,
      skipped,
      rating,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to submit feedback');
  }

  return response.json();
};

// Get feedback history
export const getFeedbackHistory = async (cookedOnly = false, skippedOnly = false) => {
  const params = new URLSearchParams();
  if (cookedOnly) params.append('cooked_only', 'true');
  if (skippedOnly) params.append('skipped_only', 'true');

  const response = await fetch(`${API_URL}/feedback/history?${params}`, {
    headers: {
      ...getAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error('Failed to get feedback history');
  }

  return response.json();
};

// Delete feedback (un-skip or un-cook a recipe)
export const deleteFeedback = async (recipeId) => {
  const response = await fetch(`${API_URL}/feedback/${recipeId}`, {
    method: 'DELETE',
    headers: {
      ...getAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error('Failed to delete feedback');
  }

  return response.json();
};
