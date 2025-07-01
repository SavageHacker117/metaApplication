document.getElementById('proposeBtn').onclick = function() {
  let title = document.getElementById('lawTitle').value;
  let text = document.getElementById('lawText').value;
  if(window.rubyEngine) window.rubyEngine.plugins.city_constitution.propose_law(title, text);
};
document.getElementById('voteBtn').onclick = function() {
  if(window.rubyEngine) window.rubyEngine.plugins.city_constitution.vote_law(0, true);
};
document.getElementById('enforceBtn').onclick = function() {
  if(window.rubyEngine) window.rubyEngine.plugins.city_constitution.enforce_law(0);
};
