(function () {
  const page = document.body.dataset.page || "";
  document.querySelectorAll(".nav a[data-page]").forEach((link) => {
    if (link.dataset.page === page) {
      link.classList.add("active");
    }
  });
})();
