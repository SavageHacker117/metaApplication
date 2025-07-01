document.getElementById('pulseBtn').onclick = function() {
  if(window.rubyEngine) {
    let metrics = window.rubyEngine.plugins.world_pulse_monitor.get_pulse();
    document.getElementById('pulseLog').innerHTML = "<pre>"+JSON.stringify(metrics,null,2)+"</pre>";
  }
};
