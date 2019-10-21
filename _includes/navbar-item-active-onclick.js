$(function() {
    $('a[href*="#"]').on('click', function(e) {
        // e.preventDefault()
        oldObjChild = $('.active > a'); // gets active nav-item child nav-link
        oldObj = $('.active'); // gets the active nav-item
        oldObj.removeClass('active'); // remove active from old nav-item
        $(this).parent().addClass('active'); // set the active class on the nav-item that called the function
    });
});
