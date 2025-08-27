window.MathJax = {
  chtml: {
    fontURL: null,  // Disable web fonts
    mathmlSpacing: false
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      // Force sans-serif after MathJax initializes
      const style = document.createElement('style');
      style.textContent = `
        mjx-container, mjx-container * {
          font-family: sans-serif !important;
        }
      `;
      document.head.appendChild(style);
    }
  }
};